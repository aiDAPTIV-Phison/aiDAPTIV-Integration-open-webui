import time
import logging
import sys
import os
import base64

import asyncio
from aiocache import cached
from typing import Any, Optional
import random
import json
import html
import inspect
import re
import ast

from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor


from fastapi import Request, HTTPException
from starlette.responses import Response, StreamingResponse, JSONResponse


from open_webui.models.chats import Chats
from open_webui.models.folders import Folders
from open_webui.models.users import Users
from open_webui.socket.main import (
    get_event_call,
    get_event_emitter,
    get_active_status_by_user_id,
)
from open_webui.routers.tasks import (
    generate_queries,
    generate_title,
    generate_follow_ups,
    generate_image_prompt,
    generate_chat_tags,
)
from open_webui.routers.files import get_file_by_id
from open_webui.routers.retrieval import process_web_search, SearchForm
from open_webui.routers.images import (
    load_b64_image_data,
    image_generations,
    GenerateImageForm,
    upload_image,
)
from open_webui.routers.pipelines import (
    process_pipeline_inlet_filter,
    process_pipeline_outlet_filter,
)
from open_webui.routers.memories import query_memory, QueryMemoryForm

from open_webui.utils.webhook import post_webhook


from open_webui.models.users import UserModel
from open_webui.models.functions import Functions
from open_webui.models.models import Models

from open_webui.retrieval.utils import get_sources_from_items


from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.task import (
    get_task_model_id,
    rag_template,
    tools_function_calling_generation_template,
)
from open_webui.utils.misc import (
    deep_update,
    get_message_list,
    add_or_update_system_message,
    add_or_update_user_message,
    get_last_user_message,
    get_last_assistant_message,
    prepend_to_first_user_message_content,
    convert_logit_bias_input_to_json,
)
from open_webui.utils.tools import get_tools
from open_webui.utils.plugin import load_function_module_by_id
from open_webui.utils.filter import (
    get_sorted_filter_ids,
    process_filter_functions,
)
from open_webui.utils.code_interpreter import execute_code_jupyter
from open_webui.utils.payload import apply_model_system_prompt_to_body

from open_webui.tasks import create_task

from open_webui.config import (
    CACHE_DIR,
    DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE,
    DEFAULT_CODE_INTERPRETER_PROMPT,
    RAG_SELF_KM
)
from open_webui.env import (
    SRC_LOG_LEVELS,
    GLOBAL_LOG_LEVEL,
    CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE,
    BYPASS_MODEL_ACCESS_CONTROL,
    ENABLE_REALTIME_CHAT_SAVE,
)
from open_webui.constants import TASKS


logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


def clean_timing_info_from_content(content):
    """
    從內容中移除計時資訊，避免影響後續的 LLM 對話

    Args:
        content (str): 包含計時資訊的內容

    Returns:
        str: 清理後的內容
    """
    if not content:
        return content

    # 移除 Time to first token 行，保留原有的換行結構
    content = re.sub(r'^Time to first token: \d+\.?\d* s\s*\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'\nTime to first token: \d+\.?\d* s\s*\n?', '\n', content, flags=re.MULTILINE)

    content = re.sub(r'^Time to first token: \d+\.?\d* s\s*\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'\nTime to first token: \d+\.?\d* s\s*\n?', '\n', content, flags=re.MULTILINE)
    # 移除 emoji 版本的 Time to first token 行
    # 標題格式: ### 🟢Time to first token: X.XX s
    content = re.sub(r'^###\s*🟢\s*Time to first token: \d+\.?\d* s\s*\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n###\s*🟢\s*Time to first token: \d+\.?\d* s\s*\n?', '\n', content, flags=re.MULTILINE)

    # 移除 Total Time 行，保留原有的換行結構
    content = re.sub(r'^Total Time: \d+\.?\d* s\s*\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'\nTotal Time: \d+\.?\d* s\s*\n?', '\n', content, flags=re.MULTILINE)

    # 移除 Token Rate 行，保留原有的換行結構
    content = re.sub(r'^Token Rate: \d+\.?\d* tokens\/s\s*\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'\nToken Rate: \d+\.?\d* tokens\/s\s*\n?', '\n', content, flags=re.MULTILINE)

    # 清理首尾空白
    content = content.strip()

    return content


def clean_timing_info_from_messages(messages):
    """
    從訊息列表中移除計時資訊

    Args:
        messages (list): 訊息列表

    Returns:
        list: 清理後的訊息列表
    """
    cleaned_messages = []

    for message in messages:
        cleaned_message = message.copy()

        if isinstance(cleaned_message.get("content"), str):
            cleaned_message["content"] = clean_timing_info_from_content(cleaned_message["content"])
        elif isinstance(cleaned_message.get("content"), list):
            # 處理 content 為列表的情況（可能包含多個區塊）
            cleaned_content = []
            for item in cleaned_message["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    cleaned_item = item.copy()
                    cleaned_item["text"] = clean_timing_info_from_content(item.get("text", ""))
                    cleaned_content.append(cleaned_item)
                else:
                    cleaned_content.append(item)
            cleaned_message["content"] = cleaned_content

        cleaned_messages.append(cleaned_message)

    return cleaned_messages


async def chat_completion_tools_handler(
    request: Request, body: dict, extra_params: dict, user: UserModel, models, tools
) -> tuple[dict, dict]:
    async def get_content_from_response(response) -> Optional[str]:
        content = None
        if hasattr(response, "body_iterator"):
            async for chunk in response.body_iterator:
                data = json.loads(chunk.decode("utf-8"))
                content = data["choices"][0]["message"]["content"]

            # Cleanup any remaining background tasks if necessary
            if response.background is not None:
                await response.background()
        else:
            content = response["choices"][0]["message"]["content"]
        return content

    def get_tools_function_calling_payload(messages, task_model_id, content):
        # 移除計時資訊，避免影響工具函式的提示生成
        try:
            messages = clean_timing_info_from_messages(messages)
        except Exception:
            pass
        user_message = get_last_user_message(messages)
        history = "\n".join(
            f"{message['role'].upper()}: \"\"\"{message['content']}\"\"\""
            for message in messages[::-1][:4]
        )

        prompt = f"History:\n{history}\nQuery: {user_message}"

        return {
            "model": task_model_id,
            "messages": [
                {"role": "system", "content": content},
                {"role": "user", "content": f"Query: {prompt}"},
            ],
            "stream": False,
            "metadata": {"task": str(TASKS.FUNCTION_CALLING)},
        }

    event_caller = extra_params["__event_call__"]
    metadata = extra_params["__metadata__"]

    task_model_id = get_task_model_id(
        body["model"],
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )

    skip_files = False
    sources = []

    specs = [tool["spec"] for tool in tools.values()]
    tools_specs = json.dumps(specs)

    if request.app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE != "":
        template = request.app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE
    else:
        template = DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE

    tools_function_calling_prompt = tools_function_calling_generation_template(
        template, tools_specs
    )
    payload = get_tools_function_calling_payload(
        body["messages"], task_model_id, tools_function_calling_prompt
    )

    try:
        response = await generate_chat_completion(request, form_data=payload, user=user)
        log.debug(f"{response=}")
        content = await get_content_from_response(response)
        log.debug(f"{content=}")

        if not content:
            return body, {}

        try:
            content = content[content.find("{") : content.rfind("}") + 1]
            if not content:
                raise Exception("No JSON object found in the response")

            result = json.loads(content)

            async def tool_call_handler(tool_call):
                nonlocal skip_files

                log.debug(f"{tool_call=}")

                tool_function_name = tool_call.get("name", None)
                if tool_function_name not in tools:
                    return body, {}

                tool_function_params = tool_call.get("parameters", {})

                try:
                    tool = tools[tool_function_name]

                    spec = tool.get("spec", {})
                    allowed_params = (
                        spec.get("parameters", {}).get("properties", {}).keys()
                    )
                    tool_function_params = {
                        k: v
                        for k, v in tool_function_params.items()
                        if k in allowed_params
                    }

                    if tool.get("direct", False):
                        tool_result = await event_caller(
                            {
                                "type": "execute:tool",
                                "data": {
                                    "id": str(uuid4()),
                                    "name": tool_function_name,
                                    "params": tool_function_params,
                                    "server": tool.get("server", {}),
                                    "session_id": metadata.get("session_id", None),
                                },
                            }
                        )
                    else:
                        tool_function = tool["callable"]
                        tool_result = await tool_function(**tool_function_params)

                except Exception as e:
                    tool_result = str(e)

                tool_result_files = []
                if isinstance(tool_result, list):
                    for item in tool_result:
                        # check if string
                        if isinstance(item, str) and item.startswith("data:"):
                            tool_result_files.append(item)
                            tool_result.remove(item)

                if isinstance(tool_result, dict) or isinstance(tool_result, list):
                    tool_result = json.dumps(tool_result, indent=2)

                if isinstance(tool_result, str):
                    tool = tools[tool_function_name]
                    tool_id = tool.get("tool_id", "")

                    tool_name = (
                        f"{tool_id}/{tool_function_name}"
                        if tool_id
                        else f"{tool_function_name}"
                    )

                    # Citation is enabled for this tool
                    sources.append(
                        {
                            "source": {
                                "name": (f"TOOL:{tool_name}"),
                            },
                            "document": [tool_result],
                            "metadata": [
                                {
                                    "source": (f"TOOL:{tool_name}"),
                                    "parameters": tool_function_params,
                                }
                            ],
                            "tool_result": True,
                        }
                    )
                    # Citation is not enabled for this tool
                    body["messages"] = add_or_update_user_message(
                        f"\nTool `{tool_name}` Output: {tool_result}",
                        body["messages"],
                    )

                    if (
                        tools[tool_function_name]
                        .get("metadata", {})
                        .get("file_handler", False)
                    ):
                        skip_files = True

            # check if "tool_calls" in result
            if result.get("tool_calls"):
                for tool_call in result.get("tool_calls"):
                    await tool_call_handler(tool_call)
            else:
                await tool_call_handler(result)

        except Exception as e:
            log.debug(f"Error: {e}")
            content = None
    except Exception as e:
        log.debug(f"Error: {e}")
        content = None

    log.debug(f"tool_contexts: {sources}")

    if skip_files and "files" in body.get("metadata", {}):
        del body["metadata"]["files"]

    return body, {"sources": sources}


async def chat_memory_handler(
    request: Request, form_data: dict, extra_params: dict, user
):
    try:
        results = await query_memory(
            request,
            QueryMemoryForm(
                **{
                    "content": get_last_user_message(form_data["messages"]) or "",
                    "k": 3,
                }
            ),
            user,
        )
    except Exception as e:
        log.debug(e)
        results = None

    user_context = ""
    if results and hasattr(results, "documents"):
        if results.documents and len(results.documents) > 0:
            for doc_idx, doc in enumerate(results.documents[0]):
                created_at_date = "Unknown Date"

                if results.metadatas[0][doc_idx].get("created_at"):
                    created_at_timestamp = results.metadatas[0][doc_idx]["created_at"]
                    created_at_date = time.strftime(
                        "%Y-%m-%d", time.localtime(created_at_timestamp)
                    )

                user_context += f"{doc_idx + 1}. [{created_at_date}] {doc}\n"

    form_data["messages"] = add_or_update_system_message(
        f"User Context:\n{user_context}\n", form_data["messages"], append=True
    )

    return form_data


async def chat_web_search_handler(
    request: Request, form_data: dict, extra_params: dict, user
):
    event_emitter = extra_params["__event_emitter__"]
    await event_emitter(
        {
            "type": "status",
            "data": {
                "action": "web_search",
                "description": "Generating search query",
                "done": False,
            },
        }
    )

    messages = form_data["messages"]
    user_message = get_last_user_message(messages)

    queries = []
    try:
        res = await generate_queries(
            request,
            {
                "model": form_data["model"],
                "messages": messages,
                "prompt": user_message,
                "type": "web_search",
            },
            user,
        )

        response = res["choices"][0]["message"]["content"]

        try:
            bracket_start = response.find("{")
            bracket_end = response.rfind("}") + 1

            if bracket_start == -1 or bracket_end == -1:
                raise Exception("No JSON object found in the response")

            response = response[bracket_start:bracket_end]
            queries = json.loads(response)
            queries = queries.get("queries", [])
        except Exception as e:
            queries = [response]

    except Exception as e:
        log.exception(e)
        queries = [user_message]

    # Check if generated queries are empty
    if len(queries) == 1 and queries[0].strip() == "":
        queries = [user_message]

    # Check if queries are not found
    if len(queries) == 0:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "action": "web_search",
                    "description": "No search query generated",
                    "done": True,
                },
            }
        )
        return form_data

    await event_emitter(
        {
            "type": "status",
            "data": {
                "action": "web_search",
                "description": "Searching the web",
                "done": False,
            },
        }
    )

    try:
        results = await process_web_search(
            request,
            SearchForm(queries=queries),
            user=user,
        )

        if results:
            files = form_data.get("files", [])

            if results.get("collection_names"):
                for col_idx, collection_name in enumerate(
                    results.get("collection_names")
                ):
                    files.append(
                        {
                            "collection_name": collection_name,
                            "name": ", ".join(queries),
                            "type": "web_search",
                            "urls": results["filenames"],
                            "queries": queries,
                        }
                    )
            elif results.get("docs"):
                # Invoked when bypass embedding and retrieval is set to True
                docs = results["docs"]
                files.append(
                    {
                        "docs": docs,
                        "name": ", ".join(queries),
                        "type": "web_search",
                        "urls": results["filenames"],
                        "queries": queries,
                    }
                )

            form_data["files"] = files

            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "action": "web_search",
                        "description": "Searched {{count}} sites",
                        "urls": results["filenames"],
                        "done": True,
                    },
                }
            )
        else:
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "action": "web_search",
                        "description": "No search results found",
                        "done": True,
                        "error": True,
                    },
                }
            )

    except Exception as e:
        log.exception(e)
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "action": "web_search",
                    "description": "An error occurred while searching the web",
                    "queries": queries,
                    "done": True,
                    "error": True,
                },
            }
        )

    return form_data


async def chat_image_generation_handler(
    request: Request, form_data: dict, extra_params: dict, user
):
    __event_emitter__ = extra_params["__event_emitter__"]
    await __event_emitter__(
        {
            "type": "status",
            "data": {"description": "Generating an image", "done": False},
        }
    )

    messages = form_data["messages"]
    user_message = get_last_user_message(messages)

    prompt = user_message
    negative_prompt = ""

    if request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION:
        try:
            res = await generate_image_prompt(
                request,
                {
                    "model": form_data["model"],
                    "messages": messages,
                },
                user,
            )

            response = res["choices"][0]["message"]["content"]

            try:
                bracket_start = response.find("{")
                bracket_end = response.rfind("}") + 1

                if bracket_start == -1 or bracket_end == -1:
                    raise Exception("No JSON object found in the response")

                response = response[bracket_start:bracket_end]
                response = json.loads(response)
                prompt = response.get("prompt", [])
            except Exception as e:
                prompt = user_message

        except Exception as e:
            log.exception(e)
            prompt = user_message

    system_message_content = ""

    try:
        images = await image_generations(
            request=request,
            form_data=GenerateImageForm(**{"prompt": prompt}),
            user=user,
        )

        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Generated an image", "done": True},
            }
        )

        await __event_emitter__(
            {
                "type": "files",
                "data": {
                    "files": [
                        {
                            "type": "image",
                            "url": image["url"],
                        }
                        for image in images
                    ]
                },
            }
        )

        system_message_content = "<context>User is shown the generated image, tell the user that the image has been generated</context>"
    except Exception as e:
        log.exception(e)
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"An error occurred while generating an image",
                    "done": True,
                },
            }
        )

        system_message_content = "<context>Unable to generate an image, tell the user that an error occurred</context>"

    if system_message_content:
        form_data["messages"] = add_or_update_system_message(
            system_message_content, form_data["messages"]
        )

    return form_data


async def chat_completion_files_handler(
    request: Request, body: dict, user: UserModel
) -> tuple[dict, dict[str, list]]:
    sources = []
    metadata = body.get("metadata", {})

    if files := metadata.get("files", None):
        queries = []
        if not RAG_SELF_KM:
            try:
                # 第一次 LLM 調用：生成查詢
                query_gen_start = time.time()
                log.debug(f"[TTFT] Starting query generation (first LLM call)")

                queries_response = await generate_queries(
                    request,
                    {
                        "model": body["model"],
                        "messages": body["messages"],
                        "type": "retrieval",
                    },
                    user,
                )
                queries_response = queries_response["choices"][0]["message"]["content"]

                query_gen_time = round(time.time() - query_gen_start, 2)
                log.debug(f"[TTFT] Query generation completed in {query_gen_time}s")

                try:
                    bracket_start = queries_response.find("{")
                    bracket_end = queries_response.rfind("}") + 1

                    if bracket_start == -1 or bracket_end == -1:
                        raise Exception("No JSON object found in the response")

                    queries_response = queries_response[bracket_start:bracket_end]
                    queries_response = json.loads(queries_response)
                except Exception as e:
                    queries_response = {"queries": [queries_response]}

                queries = queries_response.get("queries", [])
            except:
                pass

        if len(queries) == 0:
            queries = [get_last_user_message(body["messages"])]

        try:
            # 執行檢索
            retrieval_start = time.time()
            log.debug(f"[TTFT] Starting document retrieval with queries: {queries}")

            # Offload get_sources_from_items to a separate thread
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as executor:
                sources = await loop.run_in_executor(
                    executor,
                    lambda: get_sources_from_items(
                        request=request,
                        items=files,
                        queries=queries,
                        embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                            query, prefix=prefix, user=user
                        ),
                        k=request.app.state.config.TOP_K,
                        reranking_function=(
                            (
                                lambda sentences: request.app.state.RERANKING_FUNCTION(
                                    sentences, user=user
                                )
                            )
                            if request.app.state.RERANKING_FUNCTION
                            else None
                        ),
                        k_reranker=request.app.state.config.TOP_K_RERANKER,
                        r=request.app.state.config.RELEVANCE_THRESHOLD,
                        hybrid_bm25_weight=request.app.state.config.HYBRID_BM25_WEIGHT,
                        hybrid_search=request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
                        full_context=request.app.state.config.RAG_FULL_CONTEXT,
                        user=user,
                    ),
                )

            retrieval_time = round(time.time() - retrieval_start, 2)
            log.info(f"[TTFT] Document retrieval completed in {retrieval_time}s")

            # 重置 TTFT 起點：排除查詢生成和檢索的時間
            # 只計算第二次 LLM 調用（生成答案）的時間
            answer_gen_start = time.time()
            metadata['__ttft_start_time__'] = answer_gen_start
            body['metadata'] = metadata
            log.info(f"[TTFT] Reset TTFT start time to {answer_gen_start} (after query generation and retrieval)")

        except Exception as e:
            log.exception(e)

        log.debug(f"rag_contexts:sources: {sources}")

    return body, {"sources": sources}


def apply_params_to_form_data(form_data, model):
    params = form_data.pop("params", {})
    custom_params = params.pop("custom_params", {})

    open_webui_params = {
        "stream_response": bool,
        "stream_delta_chunk_size": int,
        "function_calling": str,
        "system": str,
    }

    for key in list(params.keys()):
        if key in open_webui_params:
            del params[key]

    if custom_params:
        # Attempt to parse custom_params if they are strings
        for key, value in custom_params.items():
            if isinstance(value, str):
                try:
                    # Attempt to parse the string as JSON
                    custom_params[key] = json.loads(value)
                except json.JSONDecodeError:
                    # If it fails, keep the original string
                    pass

        # If custom_params are provided, merge them into params
        params = deep_update(params, custom_params)

    if model.get("owned_by") == "ollama":
        # Ollama specific parameters
        form_data["options"] = params
    else:
        if isinstance(params, dict):
            for key, value in params.items():
                if value is not None:
                    form_data[key] = value

        if "logit_bias" in params and params["logit_bias"] is not None:
            try:
                form_data["logit_bias"] = json.loads(
                    convert_logit_bias_input_to_json(params["logit_bias"])
                )
            except Exception as e:
                log.exception(f"Error parsing logit_bias: {e}")

    return form_data


async def process_chat_payload(request, form_data, user, metadata, model):
    # Pipeline Inlet -> Filter Inlet -> Chat Memory -> Chat Web Search -> Chat Image Generation
    # -> Chat Code Interpreter (Form Data Update) -> (Default) Chat Tools Function Calling
    # -> Chat Files

    # 記錄請求開始時間（這是 TTFT 的真正起點）
    request_start_time = time.time()
    metadata['__ttft_start_time__'] = request_start_time  # 將時間傳遞給 response handler
    log.info(f"[TTFT] Chat request started (process_chat_payload) at {request_start_time}")

    form_data = apply_params_to_form_data(form_data, model)
    # 立刻清理進入 pipeline 前的使用者訊息，避免一開始就把含計時的訊息流入各種下游呼叫
    try:
        if isinstance(form_data, dict) and isinstance(form_data.get("messages"), list):
            metadata.setdefault("__messages_with_timing__", form_data["messages"])
            form_data["messages"] = clean_timing_info_from_messages(form_data["messages"])
    except Exception:
        pass
    log.debug(f"form_data: {form_data}")

    event_emitter = get_event_emitter(metadata)
    event_call = get_event_call(metadata)

    extra_params = {
        "__event_emitter__": event_emitter,
        "__event_call__": event_call,
        "__user__": user.model_dump() if isinstance(user, UserModel) else {},
        "__metadata__": metadata,
        "__request__": request,
        "__model__": model,
    }

    # Initialize events to store additional event to be sent to the client
    # Initialize contexts and citation
    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    task_model_id = get_task_model_id(
        form_data["model"],
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )

    events = []
    sources = []

    # Folder "Project" handling
    # Check if the request has chat_id and is inside of a folder
    chat_id = metadata.get("chat_id", None)
    if chat_id and user:
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user.id)
        if chat and chat.folder_id:
            folder = Folders.get_folder_by_id_and_user_id(chat.folder_id, user.id)

            if folder and folder.data:
                if "system_prompt" in folder.data:
                    form_data = apply_model_system_prompt_to_body(
                        folder.data["system_prompt"], form_data, metadata, user
                    )
                if "files" in folder.data:
                    form_data["files"] = [
                        *folder.data["files"],
                        *form_data.get("files", []),
                    ]

    # Model "Knowledge" handling
    user_message = get_last_user_message(form_data["messages"])
    model_knowledge = model.get("info", {}).get("meta", {}).get("knowledge", False)

    if model_knowledge:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "action": "knowledge_search",
                    "query": user_message,
                    "done": False,
                },
            }
        )

        knowledge_files = []
        for item in model_knowledge:
            if item.get("collection_name"):
                knowledge_files.append(
                    {
                        "id": item.get("collection_name"),
                        "name": item.get("name"),
                        "legacy": True,
                    }
                )
            elif item.get("collection_names"):
                knowledge_files.append(
                    {
                        "name": item.get("name"),
                        "type": "collection",
                        "collection_names": item.get("collection_names"),
                        "legacy": True,
                    }
                )
            else:
                knowledge_files.append(item)

        files = form_data.get("files", [])
        files.extend(knowledge_files)
        form_data["files"] = files

    variables = form_data.pop("variables", None)

    # Process the form_data through the pipeline
    try:
        form_data = await process_pipeline_inlet_filter(
            request, form_data, user, models
        )
    except Exception as e:
        raise e

    try:
        filter_functions = [
            Functions.get_function_by_id(filter_id)
            for filter_id in get_sorted_filter_ids(
                request, model, metadata.get("filter_ids", [])
            )
        ]

        form_data, flags = await process_filter_functions(
            request=request,
            filter_functions=filter_functions,
            filter_type="inlet",
            form_data=form_data,
            extra_params=extra_params,
        )
    except Exception as e:
        raise Exception(f"Error: {e}")

    features = form_data.pop("features", None)
    if features:
        if "memory" in features and features["memory"]:
            form_data = await chat_memory_handler(
                request, form_data, extra_params, user
            )

        if "web_search" in features and features["web_search"]:
            form_data = await chat_web_search_handler(
                request, form_data, extra_params, user
            )

        if "image_generation" in features and features["image_generation"]:
            form_data = await chat_image_generation_handler(
                request, form_data, extra_params, user
            )

        if "code_interpreter" in features and features["code_interpreter"]:
            form_data["messages"] = add_or_update_user_message(
                (
                    request.app.state.config.CODE_INTERPRETER_PROMPT_TEMPLATE
                    if request.app.state.config.CODE_INTERPRETER_PROMPT_TEMPLATE != ""
                    else DEFAULT_CODE_INTERPRETER_PROMPT
                ),
                form_data["messages"],
            )

    tool_ids = form_data.pop("tool_ids", None)
    files = form_data.pop("files", None)

    # Remove files duplicates
    if files:
        files = list({json.dumps(f, sort_keys=True): f for f in files}.values())

    metadata = {
        **metadata,
        "tool_ids": tool_ids,
        "files": files,
    }
    form_data["metadata"] = metadata

    # Server side tools
    tool_ids = metadata.get("tool_ids", None)
    # Client side tools
    tool_servers = metadata.get("tool_servers", None)

    log.debug(f"{tool_ids=}")
    log.debug(f"{tool_servers=}")

    tools_dict = {}

    if tool_ids:
        # 只用# collection時未帶, 為None
        tools_dict = get_tools(
            request,
            tool_ids,
            user,
            {
                **extra_params,
                "__model__": models[task_model_id],
                "__messages__": form_data["messages"],
                "__files__": metadata.get("files", []),
            },
        )

    if tool_servers:
        for tool_server in tool_servers:
            tool_specs = tool_server.pop("specs", [])

            for tool in tool_specs:
                tools_dict[tool["name"]] = {
                    "spec": tool,
                    "direct": True,
                    "server": tool_server,
                }

    if tools_dict:
        if metadata.get("params", {}).get("function_calling") == "native":
            # If the function calling is native, then call the tools function calling handler
            metadata["tools"] = tools_dict
            form_data["tools"] = [
                {"type": "function", "function": tool.get("spec", {})}
                for tool in tools_dict.values()
            ]
        else:
            # If the function calling is not native, then call the tools function calling handler
            try:
                form_data, flags = await chat_completion_tools_handler(
                    request, form_data, extra_params, user, models, tools_dict
                )
                sources.extend(flags.get("sources", []))
            except Exception as e:
                log.exception(e)

    try:
        form_data, flags = await chat_completion_files_handler(request, form_data, user)
        sources.extend(flags.get("sources", []))
    except Exception as e:
        log.exception(e)

    # If context is not empty, insert it into the messages
    if len(sources) > 0:
        context_string = ""
        citation_idx_map = {}
        # 當 RAG_SELF_KM 為 True 時，只提取純文字內容，不包含 <source> 標籤
        if RAG_SELF_KM:
            # Agent Builder 格式：只包含純文字內容，用換行符分隔
            relevance_source_index, _ = max(enumerate(sources), key=lambda x: x[1].get("distances", [0])[0])
            sources = [sources[relevance_source_index]]
            for source in sources:
                is_tool_result = source.get("tool_result", False)

                if "document" in source and not is_tool_result:
                    for document_text in source["document"]:
                        if document_text:
                            context_string += f"{document_text}\n"
            context_string = context_string.strip()
        else:
            # 原本做法：使用 <source> 標籤包裝
            for source in sources:
                is_tool_result = source.get("tool_result", False)
                if "document" in source and not is_tool_result:
                    for document_text, document_metadata in zip(
                        source["document"], source["metadata"]
                    ):
                        source_name = source.get("source", {}).get("name", None)
                        source_id = (
                            document_metadata.get("source", None)
                            or source.get("source", {}).get("id", None)
                            or "N/A"
                        )

                        if source_id not in citation_idx_map:
                            citation_idx_map[source_id] = len(citation_idx_map) + 1

                        context_string += (
                            f'<source id="{citation_idx_map[source_id]}"'
                            + (f' name="{source_name}"' if source_name else "")
                            + f">{document_text}</source>\n"
                        )
            context_string = context_string.strip()


        # 新作法（健壯化）：選擇最相關的來源，優先使用已提供的 document 文本
        # try:
        #     # 從各來源的 distances 中選最大（若無 distances 則用 0）
        #     distances_with_index = []
        #     for idx, s in enumerate(sources):
        #         dists = s.get("distances", [])
        #         score = dists[0] if isinstance(dists, list) and len(dists) > 0 else 0
        #         distances_with_index.append((idx, score))

        #     relevance_source_index = max(distances_with_index, key=lambda x: x[1])[0] if distances_with_index else 0
        # except Exception:
        #     relevance_source_index = 0

        # selected_source = sources[relevance_source_index] if len(sources) > 0 else {}
        # source_name = selected_source.get("source", {}).get("name", None)

        # # 優先使用 KM RAG API 已提供的 document 文字，避免誤把 collection_id 當成 file_id 去查檔案
        # document_text = ""
        # try:
        #     docs = selected_source.get("document", [])
        #     if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], str):
        #         document_text = docs[0]
        # except Exception:
        #     document_text = ""

        # # 若沒有 document，嘗試（可選）從檔案系統補救；但避免把 collection_id 當成 file_id
        # if not document_text:
        #     meta_list = selected_source.get("metadata", [])
        #     file_id = ""
        #     if isinstance(meta_list, list) and len(meta_list) > 0:
        #         file_id = meta_list[0].get("file_id", "")
        #     # 僅當確實有 file_id 時才嘗試查檔案
        #     if file_id:
        #         try:
        #             file = await get_file_by_id(id=file_id, user=user)
        #             if file and getattr(file, "data", None):
        #                 document_text = file.data.get("content", "")
        #         except Exception:
        #             document_text = document_text or ""

        # context_string = (
        #     (f'<source name="{source_name}' if source_name else "")
        #     + f">{document_text}</source>\n"
        # )

        prompt = get_last_user_message(form_data["messages"])
        if prompt is None:
            raise Exception("No user message found")

        if context_string == "":
            if request.app.state.config.RELEVANCE_THRESHOLD == 0:
                log.debug(
                    f"With a 0 relevancy threshold for RAG, the context cannot be empty"
                )
        else:
            # Workaround for Ollama 2.0+ system prompt issue
            # TODO: replace with add_or_update_system_message
            if model.get("owned_by") == "ollama":
                form_data["messages"] = prepend_to_first_user_message_content(
                    rag_template(
                        request.app.state.config.RAG_TEMPLATE, context_string, prompt
                    ),
                    form_data["messages"],
                )
            else:
                form_data["messages"] = add_or_update_system_message(
                    rag_template(
                        request.app.state.config.RAG_TEMPLATE, context_string, prompt
                    ),
                    form_data["messages"],
                )

    # If there are citations, add them to the data_items
    sources = [
        source
        for source in sources
        if source.get("source", {}).get("name", "")
        or source.get("source", {}).get("id", "")
    ]

    if len(sources) > 0:
        events.append({"sources": sources})

    if model_knowledge:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "action": "knowledge_search",
                    "query": user_message,
                    "done": True,
                    "hidden": True,
                },
            }
        )

    # 在返回前：為 LLM 呼叫準備乾淨版的 messages（移除計時資訊），
    # 但保留原始含計時資訊版本於 metadata 以供顯示用途
    try:
        original_messages = form_data.get("messages", [])
        metadata.setdefault("__messages_with_timing__", original_messages)
        form_data["messages"] = clean_timing_info_from_messages(original_messages)
        log.info(f"[TIMING CLEANUP] Final cleanup applied to {len(form_data['messages'])} messages")
    except Exception as _e:
        # 安全失敗：若清理失敗，沿用原始 messages
        log.warning(f"[TIMING CLEANUP] Failed to clean messages: {_e}")
        pass

    return form_data, metadata, events


async def process_chat_response(
    request, response, form_data, user, metadata, model, events, tasks
):
    # 記錄 response 處理開始時間
    response_process_start_time = time.time()
    log.info(f"[TTFT] process_chat_response started at {response_process_start_time}")

    async def background_tasks_handler():
        message_map = Chats.get_messages_by_chat_id(metadata["chat_id"])
        message = message_map.get(metadata["message_id"]) if message_map else None

        if message:
            message_list = get_message_list(message_map, metadata["message_id"])

            # Remove details tags and files from the messages.
            # as get_message_list creates a new list, it does not affect
            # the original messages outside of this handler

            messages = []
            for message in message_list:
                content = message.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            content = item["text"]
                            break

                if isinstance(content, str):
                    content = re.sub(
                        r"<details\b[^>]*>.*?<\/details>|!\[.*?\]\(.*?\)",
                        "",
                        content,
                        flags=re.S | re.I,
                    ).strip()

                messages.append(
                    {
                        **message,
                        "role": message.get(
                            "role", "assistant"
                        ),  # Safe fallback for missing role
                        "content": content,
                    }
                )

            if tasks and messages:
                if (
                    TASKS.FOLLOW_UP_GENERATION in tasks
                    and tasks[TASKS.FOLLOW_UP_GENERATION]
                ):
                    res = await generate_follow_ups(
                        request,
                        {
                            "model": message["model"],
                            "messages": messages,
                            "message_id": metadata["message_id"],
                            "chat_id": metadata["chat_id"],
                        },
                        user,
                    )

                    if res and isinstance(res, dict):
                        if len(res.get("choices", [])) == 1:
                            follow_ups_string = (
                                res.get("choices", [])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                        else:
                            follow_ups_string = ""

                        follow_ups_string = follow_ups_string[
                            follow_ups_string.find("{") : follow_ups_string.rfind("}")
                            + 1
                        ]

                        try:
                            follow_ups = json.loads(follow_ups_string).get(
                                "follow_ups", []
                            )

                            Chats.upsert_message_to_chat_by_id_and_message_id(
                                metadata["chat_id"],
                                metadata["message_id"],
                                {
                                    "followUps": follow_ups,
                                },
                            )

                            await event_emitter(
                                {
                                    "type": "chat:message:follow_ups",
                                    "data": {
                                        "follow_ups": follow_ups,
                                    },
                                }
                            )
                        except Exception as e:
                            pass

                if TASKS.TITLE_GENERATION in tasks:
                    user_message = get_last_user_message(messages)
                    if user_message and len(user_message) > 100:
                        user_message = user_message[:100] + "..."

                    if tasks[TASKS.TITLE_GENERATION]:

                        res = await generate_title(
                            request,
                            {
                                "model": message["model"],
                                "messages": messages,
                                "chat_id": metadata["chat_id"],
                            },
                            user,
                        )

                        if res and isinstance(res, dict):
                            if len(res.get("choices", [])) == 1:
                                title_string = (
                                    res.get("choices", [])[0]
                                    .get("message", {})
                                    .get(
                                        "content", message.get("content", user_message)
                                    )
                                )
                            else:
                                title_string = ""

                            title_string = title_string[
                                title_string.find("{") : title_string.rfind("}") + 1
                            ]

                            try:
                                title = json.loads(title_string).get(
                                    "title", user_message
                                )
                            except Exception as e:
                                title = ""

                            if not title:
                                title = messages[0].get("content", user_message)

                            Chats.update_chat_title_by_id(metadata["chat_id"], title)

                            await event_emitter(
                                {
                                    "type": "chat:title",
                                    "data": title,
                                }
                            )
                    elif len(messages) == 2:
                        title = messages[0].get("content", user_message)

                        Chats.update_chat_title_by_id(metadata["chat_id"], title)

                        await event_emitter(
                            {
                                "type": "chat:title",
                                "data": message.get("content", user_message),
                            }
                        )

                if TASKS.TAGS_GENERATION in tasks and tasks[TASKS.TAGS_GENERATION]:
                    res = await generate_chat_tags(
                        request,
                        {
                            "model": message["model"],
                            "messages": messages,
                            "chat_id": metadata["chat_id"],
                        },
                        user,
                    )

                    if res and isinstance(res, dict):
                        if len(res.get("choices", [])) == 1:
                            tags_string = (
                                res.get("choices", [])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                        else:
                            tags_string = ""

                        tags_string = tags_string[
                            tags_string.find("{") : tags_string.rfind("}") + 1
                        ]

                        try:
                            tags = json.loads(tags_string).get("tags", [])
                            Chats.update_chat_tags_by_id(
                                metadata["chat_id"], tags, user
                            )

                            await event_emitter(
                                {
                                    "type": "chat:tags",
                                    "data": tags,
                                }
                            )
                        except Exception as e:
                            pass

    event_emitter = None
    event_caller = None
    if (
        "session_id" in metadata
        and metadata["session_id"]
        and "chat_id" in metadata
        and metadata["chat_id"]
        and "message_id" in metadata
        and metadata["message_id"]
    ):
        event_emitter = get_event_emitter(metadata)
        event_caller = get_event_call(metadata)

    # Non-streaming response
    if not isinstance(response, StreamingResponse):
        if event_emitter:
            if isinstance(response, dict) or isinstance(response, JSONResponse):

                if isinstance(response, JSONResponse) and isinstance(
                    response.body, bytes
                ):
                    try:
                        response_data = json.loads(response.body.decode("utf-8"))
                    except json.JSONDecodeError:
                        response_data = {"error": {"detail": "Invalid JSON response"}}
                else:
                    response_data = response

                if "error" in response_data:
                    error = response_data["error"].get("detail", response_data["error"])
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata["chat_id"],
                        metadata["message_id"],
                        {
                            "error": {"content": error},
                        },
                    )

                if "selected_model_id" in response_data:
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata["chat_id"],
                        metadata["message_id"],
                        {
                            "selectedModelId": response_data["selected_model_id"],
                        },
                    )

                choices = response_data.get("choices", [])
                if choices and choices[0].get("message", {}).get("content"):
                    content = response_data["choices"][0]["message"]["content"]

                    if content:
                        await event_emitter(
                            {
                                "type": "chat:completion",
                                "data": response_data,
                            }
                        )

                        title = Chats.get_chat_title_by_id(metadata["chat_id"])

                        await event_emitter(
                            {
                                "type": "chat:completion",
                                "data": {
                                    "done": True,
                                    "content": content,
                                    "title": title,
                                },
                            }
                        )

                        # Save message in the database
                        Chats.upsert_message_to_chat_by_id_and_message_id(
                            metadata["chat_id"],
                            metadata["message_id"],
                            {
                                "role": "assistant",
                                "content": content,
                            },
                        )

                        # Send a webhook notification if the user is not active
                        if not get_active_status_by_user_id(user.id):
                            webhook_url = Users.get_user_webhook_url_by_id(user.id)
                            if webhook_url:
                                post_webhook(
                                    request.app.state.WEBUI_NAME,
                                    webhook_url,
                                    f"{title} - {request.app.state.config.WEBUI_URL}/c/{metadata['chat_id']}\n\n{content}",
                                    {
                                        "action": "chat",
                                        "message": content,
                                        "title": title,
                                        "url": f"{request.app.state.config.WEBUI_URL}/c/{metadata['chat_id']}",
                                    },
                                )

                        await background_tasks_handler()

                if events and isinstance(events, list):
                    extra_response = {}
                    for event in events:
                        if isinstance(event, dict):
                            extra_response.update(event)
                        else:
                            extra_response[event] = True

                    response_data = {
                        **extra_response,
                        **response_data,
                    }

                if isinstance(response, dict):
                    response = response_data
                if isinstance(response, JSONResponse):
                    response = JSONResponse(
                        content=response_data,
                        headers=response.headers,
                        status_code=response.status_code,
                    )

            return response
        else:
            if events and isinstance(events, list) and isinstance(response, dict):
                extra_response = {}
                for event in events:
                    if isinstance(event, dict):
                        extra_response.update(event)
                    else:
                        extra_response[event] = True

                response = {
                    **extra_response,
                    **response,
                }

            return response

    # Non standard response
    if not any(
        content_type in response.headers["Content-Type"]
        for content_type in ["text/event-stream", "application/x-ndjson"]
    ):
        return response

    extra_params = {
        "__event_emitter__": event_emitter,
        "__event_call__": event_caller,
        "__user__": user.model_dump() if isinstance(user, UserModel) else {},
        "__metadata__": metadata,
        "__request__": request,
        "__model__": model,
    }
    filter_functions = [
        Functions.get_function_by_id(filter_id)
        for filter_id in get_sorted_filter_ids(
            request, model, metadata.get("filter_ids", [])
        )
    ]

    # Streaming response
    if event_emitter and event_caller:
        task_id = str(uuid4())  # Create a unique task ID.
        model_id = form_data.get("model", "")

        def split_content_and_whitespace(content):
            content_stripped = content.rstrip()
            original_whitespace = (
                content[len(content_stripped) :]
                if len(content) > len(content_stripped)
                else ""
            )
            return content_stripped, original_whitespace

        def is_opening_code_block(content):
            backtick_segments = content.split("```")
            # Even number of segments means the last backticks are opening a new block
            return len(backtick_segments) > 1 and len(backtick_segments) % 2 == 0

        # Handle as a background task
        async def response_handler(response, events):
            # 使用從 payload 處理開始時記錄的時間作為 TTFT 起點
            # 這樣可以包含所有的處理時間（RAG、工具調用、模型等待等）
            start_time = metadata.get('__ttft_start_time__', time.time())
            response_handler_start_time = time.time()
            first_token_received = False  # 追蹤是否已收到第一個 token
            ttft_value = 0.0  # 儲存 TTFT 值

            # 標記變量：防止重複添加時間信息
            ttft_added_to_stream = False  # 追蹤是否已在串流中添加 TTFT
            ttft_added_to_final = False  # 追蹤是否已在最終內容中添加 TTFT
            total_time_added = False  # 追蹤是否已添加 Total Time
            stream_handler_depth = 0  # 追蹤 stream_body_handler 的遞歸深度

            # 追蹤 decode throughput 相關的 metrics
            metrics_info = {}  # 儲存從 API 返回的 metrics 信息（first_token_time, last_token_time）
            usage_info = {}  # 儲存從 API 返回的 usage 信息（completion_tokens, prompt_tokens）
            
            # 備用方案：自己追蹤 token timing（如果 API 不提供）
            first_content_token_time = None  # 第一個實際內容 token 的時間
            last_content_token_time = None  # 最後一個內容 token 的時間
            total_output_tokens = 0  # 追蹤輸出的 token 數量（粗略估計）

            log.info(f"[TTFT] Response handler started at {response_handler_start_time} (request started at {start_time}, elapsed: {round(response_handler_start_time - start_time, 3)}s)")

            def serialize_content_blocks(content_blocks, raw=False):
                content = ""

                for block in content_blocks:
                    if block["type"] == "text":
                        block_content = block["content"].strip()
                        if block_content:
                            content = f"{content}{block_content}\n"
                    elif block["type"] == "tool_calls":
                        attributes = block.get("attributes", {})

                        tool_calls = block.get("content", [])
                        results = block.get("results", [])

                        if content and not content.endswith("\n"):
                            content += "\n"

                        if results:

                            tool_calls_display_content = ""
                            for tool_call in tool_calls:

                                tool_call_id = tool_call.get("id", "")
                                tool_name = tool_call.get("function", {}).get(
                                    "name", ""
                                )
                                tool_arguments = tool_call.get("function", {}).get(
                                    "arguments", ""
                                )

                                tool_result = None
                                tool_result_files = None
                                for result in results:
                                    if tool_call_id == result.get("tool_call_id", ""):
                                        tool_result = result.get("content", None)
                                        tool_result_files = result.get("files", None)
                                        break

                                if tool_result:
                                    tool_calls_display_content = f'{tool_calls_display_content}<details type="tool_calls" done="true" id="{tool_call_id}" name="{tool_name}" arguments="{html.escape(json.dumps(tool_arguments))}" result="{html.escape(json.dumps(tool_result, ensure_ascii=False))}" files="{html.escape(json.dumps(tool_result_files)) if tool_result_files else ""}">\n<summary>Tool Executed</summary>\n</details>\n'
                                else:
                                    tool_calls_display_content = f'{tool_calls_display_content}<details type="tool_calls" done="false" id="{tool_call_id}" name="{tool_name}" arguments="{html.escape(json.dumps(tool_arguments))}">\n<summary>Executing...</summary>\n</details>\n'

                            if not raw:
                                content = f"{content}{tool_calls_display_content}"
                        else:
                            tool_calls_display_content = ""

                            for tool_call in tool_calls:
                                tool_call_id = tool_call.get("id", "")
                                tool_name = tool_call.get("function", {}).get(
                                    "name", ""
                                )
                                tool_arguments = tool_call.get("function", {}).get(
                                    "arguments", ""
                                )

                                tool_calls_display_content = f'{tool_calls_display_content}\n<details type="tool_calls" done="false" id="{tool_call_id}" name="{tool_name}" arguments="{html.escape(json.dumps(tool_arguments))}">\n<summary>Executing...</summary>\n</details>\n'

                            if not raw:
                                content = f"{content}{tool_calls_display_content}"

                    elif block["type"] == "reasoning":
                        reasoning_display_content = "\n".join(
                            (f"> {line}" if not line.startswith(">") else line)
                            for line in block["content"].splitlines()
                        )

                        reasoning_duration = block.get("duration", None)

                        start_tag = block.get("start_tag", "")
                        end_tag = block.get("end_tag", "")

                        if content and not content.endswith("\n"):
                            content += "\n"

                        if reasoning_duration is not None:
                            if raw:
                                content = (
                                    f'{content}{start_tag}{block["content"]}{end_tag}\n'
                                )
                            else:
                                content = f'{content}<details type="reasoning" done="true" duration="{reasoning_duration}">\n<summary>Thought for {reasoning_duration} seconds</summary>\n{reasoning_display_content}\n</details>\n'
                        else:
                            if raw:
                                content = (
                                    f'{content}{start_tag}{block["content"]}{end_tag}\n'
                                )
                            else:
                                content = f'{content}<details type="reasoning" done="false">\n<summary>Thinking…</summary>\n{reasoning_display_content}\n</details>\n'

                    elif block["type"] == "code_interpreter":
                        attributes = block.get("attributes", {})
                        output = block.get("output", None)
                        lang = attributes.get("lang", "")

                        content_stripped, original_whitespace = (
                            split_content_and_whitespace(content)
                        )
                        if is_opening_code_block(content_stripped):
                            # Remove trailing backticks that would open a new block
                            content = (
                                content_stripped.rstrip("`").rstrip()
                                + original_whitespace
                            )
                        else:
                            # Keep content as is - either closing backticks or no backticks
                            content = content_stripped + original_whitespace

                        if content and not content.endswith("\n"):
                            content += "\n"

                        if output:
                            output = html.escape(json.dumps(output))

                            if raw:
                                content = f'{content}<code_interpreter type="code" lang="{lang}">\n{block["content"]}\n</code_interpreter>\n```output\n{output}\n```\n'
                            else:
                                content = f'{content}<details type="code_interpreter" done="true" output="{output}">\n<summary>Analyzed</summary>\n```{lang}\n{block["content"]}\n```\n</details>\n'
                        else:
                            if raw:
                                content = f'{content}<code_interpreter type="code" lang="{lang}">\n{block["content"]}\n</code_interpreter>\n'
                            else:
                                content = f'{content}<details type="code_interpreter" done="false">\n<summary>Analyzing...</summary>\n```{lang}\n{block["content"]}\n```\n</details>\n'

                    else:
                        block_content = str(block["content"]).strip()
                        if block_content:
                            content = f"{content}{block['type']}: {block_content}\n"

                return content.strip()

            def convert_content_blocks_to_messages(content_blocks, raw=False):
                messages = []

                temp_blocks = []
                for idx, block in enumerate(content_blocks):
                    if block["type"] == "tool_calls":
                        messages.append(
                            {
                                "role": "assistant",
                                "content": serialize_content_blocks(temp_blocks, raw),
                                "tool_calls": block.get("content"),
                            }
                        )

                        results = block.get("results", [])

                        for result in results:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": result["tool_call_id"],
                                    "content": result["content"],
                                }
                            )
                        temp_blocks = []
                    else:
                        temp_blocks.append(block)

                if temp_blocks:
                    content = serialize_content_blocks(temp_blocks, raw)
                    if content:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": content,
                            }
                        )

                return messages

            def tag_content_handler(content_type, tags, content, content_blocks):
                end_flag = False

                def extract_attributes(tag_content):
                    """Extract attributes from a tag if they exist."""
                    attributes = {}
                    if not tag_content:  # Ensure tag_content is not None
                        return attributes
                    # Match attributes in the format: key="value" (ignores single quotes for simplicity)
                    matches = re.findall(r'(\w+)\s*=\s*"([^"]+)"', tag_content)
                    for key, value in matches:
                        attributes[key] = value
                    return attributes

                if content_blocks[-1]["type"] == "text":
                    for start_tag, end_tag in tags:

                        start_tag_pattern = rf"{re.escape(start_tag)}"
                        if start_tag.startswith("<") and start_tag.endswith(">"):
                            # Match start tag e.g., <tag> or <tag attr="value">
                            # remove both '<' and '>' from start_tag
                            # Match start tag with attributes
                            start_tag_pattern = (
                                rf"<{re.escape(start_tag[1:-1])}(\s.*?)?>"
                            )

                        match = re.search(start_tag_pattern, content)
                        if match:
                            attr_content = (
                                match.group(1) if match.group(1) else ""
                            )  # Ensure it's not None
                            attributes = extract_attributes(
                                attr_content
                            )  # Extract attributes safely

                            # Capture everything before and after the matched tag
                            before_tag = content[
                                : match.start()
                            ]  # Content before opening tag
                            after_tag = content[
                                match.end() :
                            ]  # Content after opening tag

                            # Remove the start tag and after from the currently handling text block
                            content_blocks[-1]["content"] = content_blocks[-1][
                                "content"
                            ].replace(match.group(0) + after_tag, "")

                            if before_tag:
                                content_blocks[-1]["content"] = before_tag

                            if not content_blocks[-1]["content"]:
                                content_blocks.pop()

                            # Append the new block
                            content_blocks.append(
                                {
                                    "type": content_type,
                                    "start_tag": start_tag,
                                    "end_tag": end_tag,
                                    "attributes": attributes,
                                    "content": "",
                                    "started_at": time.time(),
                                }
                            )

                            if after_tag:
                                content_blocks[-1]["content"] = after_tag
                                tag_content_handler(
                                    content_type, tags, after_tag, content_blocks
                                )

                            break
                elif content_blocks[-1]["type"] == content_type:
                    start_tag = content_blocks[-1]["start_tag"]
                    end_tag = content_blocks[-1]["end_tag"]

                    if end_tag.startswith("<") and end_tag.endswith(">"):
                        # Match end tag e.g., </tag>
                        end_tag_pattern = rf"{re.escape(end_tag)}"
                    else:
                        # Handle cases where end_tag is just a tag name
                        end_tag_pattern = rf"{re.escape(end_tag)}"

                    # Check if the content has the end tag
                    if re.search(end_tag_pattern, content):
                        end_flag = True

                        block_content = content_blocks[-1]["content"]
                        # Strip start and end tags from the content
                        start_tag_pattern = rf"<{re.escape(start_tag)}(.*?)>"
                        block_content = re.sub(
                            start_tag_pattern, "", block_content
                        ).strip()

                        end_tag_regex = re.compile(end_tag_pattern, re.DOTALL)
                        split_content = end_tag_regex.split(block_content, maxsplit=1)

                        # Content inside the tag
                        block_content = (
                            split_content[0].strip() if split_content else ""
                        )

                        # Leftover content (everything after `</tag>`)
                        leftover_content = (
                            split_content[1].strip() if len(split_content) > 1 else ""
                        )

                        if block_content:
                            content_blocks[-1]["content"] = block_content
                            content_blocks[-1]["ended_at"] = time.time()
                            content_blocks[-1]["duration"] = int(
                                content_blocks[-1]["ended_at"]
                                - content_blocks[-1]["started_at"]
                            )

                            # Reset the content_blocks by appending a new text block
                            if content_type != "code_interpreter":
                                if leftover_content:

                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "content": leftover_content,
                                        }
                                    )
                                else:
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "content": "",
                                        }
                                    )

                        else:
                            # Remove the block if content is empty
                            content_blocks.pop()

                            if leftover_content:
                                content_blocks.append(
                                    {
                                        "type": "text",
                                        "content": leftover_content,
                                    }
                                )
                            else:
                                content_blocks.append(
                                    {
                                        "type": "text",
                                        "content": "",
                                    }
                                )

                        # Clean processed content
                        start_tag_pattern = rf"{re.escape(start_tag)}"
                        if start_tag.startswith("<") and start_tag.endswith(">"):
                            # Match start tag e.g., <tag> or <tag attr="value">
                            # remove both '<' and '>' from start_tag
                            # Match start tag with attributes
                            start_tag_pattern = (
                                rf"<{re.escape(start_tag[1:-1])}(\s.*?)?>"
                            )

                        content = re.sub(
                            rf"{start_tag_pattern}(.|\n)*?{re.escape(end_tag)}",
                            "",
                            content,
                            flags=re.DOTALL,
                        )

                return content, content_blocks, end_flag

            message = Chats.get_message_by_id_and_message_id(
                metadata["chat_id"], metadata["message_id"]
            )

            tool_calls = []

            last_assistant_message = None
            try:
                if form_data["messages"][-1]["role"] == "assistant":
                    last_assistant_message = get_last_assistant_message(
                        form_data["messages"]
                    )
            except Exception as e:
                pass

            content = (
                message.get("content", "")
                if message
                else last_assistant_message if last_assistant_message else ""
            )

            content_blocks = [
                {
                    "type": "text",
                    "content": content,
                }
            ]

            # We might want to disable this by default
            DETECT_REASONING = True
            DETECT_SOLUTION = True
            DETECT_CODE_INTERPRETER = metadata.get("features", {}).get(
                "code_interpreter", False
            )

            reasoning_tags = [
                ("<think>", "</think>"),
                ("<thinking>", "</thinking>"),
                ("<reason>", "</reason>"),
                ("<reasoning>", "</reasoning>"),
                ("<thought>", "</thought>"),
                ("<Thought>", "</Thought>"),
                ("<|begin_of_thought|>", "<|end_of_thought|>"),
                ("◁think▷", "◁/think▷"),
            ]

            code_interpreter_tags = [("<code_interpreter>", "</code_interpreter>")]

            solution_tags = [("<|begin_of_solution|>", "<|end_of_solution|>")]

            try:
                for event in events:
                    await event_emitter(
                        {
                            "type": "chat:completion",
                            "data": event,
                        }
                    )

                    # Save message in the database
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata["chat_id"],
                        metadata["message_id"],
                        {
                            **event,
                        },
                    )

                async def stream_body_handler(response, form_data):
                    nonlocal content
                    nonlocal content_blocks
                    nonlocal start_time  # 需要訪問外層的 start_time
                    nonlocal first_token_received  # 追蹤第一個 token
                    nonlocal ttft_value  # 儲存 TTFT 值

                    # 訪問標記變量
                    nonlocal ttft_added_to_stream
                    nonlocal ttft_added_to_final
                    nonlocal total_time_added
                    nonlocal stream_handler_depth

                    # 訪問 decode throughput 相關變量
                    nonlocal metrics_info
                    nonlocal usage_info
                    nonlocal first_content_token_time
                    nonlocal last_content_token_time
                    nonlocal total_output_tokens

                    # 進入函數時增加深度
                    stream_handler_depth += 1
                    current_depth = stream_handler_depth
                    log.debug(f"[TTFT] Stream body handler entered, depth={current_depth}")

                    response_tool_calls = []

                    delta_count = 0
                    delta_chunk_size = max(
                        CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE,
                        int(
                            metadata.get("params", {}).get("stream_delta_chunk_size")
                            or 1
                        ),
                    )

                    log.debug(f"[TTFT] Stream body handler started, depth={current_depth}, first_token_received={first_token_received}")
                    line_count = 0
                    raw_data_dump = []  # 收集所有原始數據以便 debug
                    async for line in response.body_iterator:
                        line = line.decode("utf-8") if isinstance(line, bytes) else line
                        data = line

                        # Skip empty lines
                        if not data.strip():
                            continue

                        # "data:" is the prefix for each event
                        if not data.startswith("data:"):
                            continue

                        # Remove the prefix
                        data = data[len("data:") :].strip()

                        # Debug: 收集前 10 行原始數據
                        line_count += 1
                        if line_count <= 10:
                            current_time = time.time()
                            elapsed = round(current_time - start_time, 3)
                            raw_data_dump.append(f"[{elapsed}s] Line #{line_count}: {data[:300] if len(data) > 300 else data}")
                            log.debug(f"[TTFT DEBUG] [{elapsed}s] Raw line #{line_count}: {data[:200] if len(data) > 200 else data}")

                        try:
                            data = json.loads(data)

                            data, _ = await process_filter_functions(
                                request=request,
                                filter_functions=filter_functions,
                                filter_type="stream",
                                form_data=data,
                                extra_params={"__body__": form_data, **extra_params},
                            )

                            if data:
                                # 收集 usage 和 timing metrics（在任何地方都嘗試收集）
                                usage = data.get("usage", {})
                                if usage:
                                    usage_info.update(usage)
                                    log.info(f"[METRICS] ✓ Collected usage at line #{line_count}: {usage}")
                                
                                # 收集 timing metrics（從頂層）
                                timing_found = False
                                if "first_token_time" in data:
                                    metrics_info["first_token_time"] = data.get("first_token_time")
                                    log.info(f"[METRICS] ✓ Collected first_token_time at line #{line_count}: {data.get('first_token_time')}")
                                    timing_found = True
                                if "last_token_time" in data:
                                    metrics_info["last_token_time"] = data.get("last_token_time")
                                    log.info(f"[METRICS] ✓ Collected last_token_time at line #{line_count}: {data.get('last_token_time')}")
                                    timing_found = True
                                
                                # 也嘗試從 metrics 子字段讀取
                                metrics = data.get("metrics", {})
                                if metrics:
                                    if "first_token_time" in metrics:
                                        metrics_info["first_token_time"] = metrics.get("first_token_time")
                                        log.info(f"[METRICS] ✓ Collected first_token_time from metrics at line #{line_count}: {metrics.get('first_token_time')}")
                                        timing_found = True
                                    if "last_token_time" in metrics:
                                        metrics_info["last_token_time"] = metrics.get("last_token_time")
                                        log.info(f"[METRICS] ✓ Collected last_token_time from metrics at line #{line_count}: {metrics.get('last_token_time')}")
                                        timing_found = True
                                
                                # Debug: 如果是前5個chunk或者找到重要數據，記錄完整的 keys
                                if line_count <= 5 or usage or timing_found:
                                    log.debug(f"[METRICS DEBUG] Chunk #{line_count} keys: {data.keys()}")

                                if "event" in data:
                                    await event_emitter(data.get("event", {}))

                                if "selected_model_id" in data:
                                    model_id = data["selected_model_id"]
                                    Chats.upsert_message_to_chat_by_id_and_message_id(
                                        metadata["chat_id"],
                                        metadata["message_id"],
                                        {
                                            "selectedModelId": model_id,
                                        },
                                    )
                                else:
                                    choices = data.get("choices", [])
                                    if not choices:
                                        error = data.get("error", {})
                                        if error:
                                            await event_emitter(
                                                {
                                                    "type": "chat:completion",
                                                    "data": {
                                                        "error": error,
                                                    },
                                                }
                                            )
                                        # usage 已經在前面收集了，這裡只需要發送 event
                                        usage = data.get("usage", {})
                                        if usage:
                                            await event_emitter(
                                                {
                                                    "type": "chat:completion",
                                                    "data": {
                                                        "usage": usage,
                                                    },
                                                }
                                            )
                                        continue

                                    delta = choices[0].get("delta", {})
                                    delta_tool_calls = delta.get("tool_calls", None)

                                    if delta_tool_calls:
                                        for delta_tool_call in delta_tool_calls:
                                            tool_call_index = delta_tool_call.get(
                                                "index"
                                            )

                                            if tool_call_index is not None:
                                                # Check if the tool call already exists
                                                current_response_tool_call = None
                                                for (
                                                    response_tool_call
                                                ) in response_tool_calls:
                                                    if (
                                                        response_tool_call.get("index")
                                                        == tool_call_index
                                                    ):
                                                        current_response_tool_call = (
                                                            response_tool_call
                                                        )
                                                        break

                                                if current_response_tool_call is None:
                                                    # Add the new tool call
                                                    delta_tool_call.setdefault(
                                                        "function", {}
                                                    )
                                                    delta_tool_call[
                                                        "function"
                                                    ].setdefault("name", "")
                                                    delta_tool_call[
                                                        "function"
                                                    ].setdefault("arguments", "")
                                                    response_tool_calls.append(
                                                        delta_tool_call
                                                    )
                                                else:
                                                    # Update the existing tool call
                                                    delta_name = delta_tool_call.get(
                                                        "function", {}
                                                    ).get("name")
                                                    delta_arguments = (
                                                        delta_tool_call.get(
                                                            "function", {}
                                                        ).get("arguments")
                                                    )

                                                    if delta_name:
                                                        current_response_tool_call[
                                                            "function"
                                                        ]["name"] += delta_name

                                                    if delta_arguments:
                                                        current_response_tool_call[
                                                            "function"
                                                        ][
                                                            "arguments"
                                                        ] += delta_arguments

                                    value = delta.get("content")

                                    reasoning_content = (
                                        delta.get("reasoning_content")
                                        or delta.get("reasoning")
                                        or delta.get("thinking")
                                    )
                                    if reasoning_content:
                                        current_time = time.time()
                                        # 如果這是第一個 token（reasoning），記錄 TTFT
                                        if not first_token_received:
                                            ttft_value = round(current_time - start_time, 2)
                                            first_token_received = True
                                            first_content_token_time = current_time
                                        
                                        # 追蹤 reasoning token
                                        last_content_token_time = current_time
                                        total_output_tokens += 1

                                        if (
                                            not content_blocks
                                            or content_blocks[-1]["type"] != "reasoning"
                                        ):
                                            reasoning_block = {
                                                "type": "reasoning",
                                                "start_tag": "<think>",
                                                "end_tag": "</think>",
                                                "attributes": {
                                                    "type": "reasoning_content"
                                                },
                                                "content": "",
                                                "started_at": time.time(),
                                            }
                                            content_blocks.append(reasoning_block)
                                        else:
                                            reasoning_block = content_blocks[-1]

                                        reasoning_block["content"] += reasoning_content

                                        data = {
                                            "content": serialize_content_blocks(
                                                content_blocks
                                            )
                                        }

                                    if value:
                                        current_time = time.time()
                                        current_elapsed = round(current_time - start_time, 3)
                                        # 如果這是第一個 token（content），記錄 TTFT
                                        # 過濾掉標籤本身（如 <think>, </think> 等）
                                        # 只有當收到實際的內容時才計算 TTFT
                                        stripped_value = value.strip()
                                        is_tag_only = (
                                            stripped_value.startswith("<") and stripped_value.endswith(">")
                                        ) or (
                                            stripped_value in ["<think>", "</think>", "<thinking>", "</thinking>",
                                                             "<reason>", "</reason>", "<reasoning>", "</reasoning>"]
                                        )

                                        log.debug(f"[TTFT DEBUG] [{current_elapsed}s] is_tag_only={is_tag_only}, stripped_value='{stripped_value}', first_token_received={first_token_received}, ttft_added_to_stream={ttft_added_to_stream}")

                                        # 確保 TTFT 只計算和添加一次
                                        if not first_token_received and stripped_value and len(stripped_value) > 0 and not is_tag_only:
                                            # 計算 TTFT（只會發生一次，因為之後 first_token_received = True）
                                            ttft_value = round(current_time - start_time, 2)
                                            first_token_received = True
                                            first_content_token_time = current_time  # 記錄第一個 content token 時間
                                            log.debug(f"[TTFT] First token (content) received, TTFT={ttft_value}s, depth={current_depth}, content_length={len(value)}, stripped_length={len(value.strip())}, content='{stripped_value[:30]}'")

                                            # 只在未添加過時才添加 TTFT 到串流（忽略 depth，因為 depth 可能在 tool calls 後重置）
                                            if not ttft_added_to_stream:
                                                value = f"### 🟢Time to first token: {ttft_value} s\n{value}"
                                                ttft_added_to_stream = True
                                                log.debug(f"[TTFT] Added TTFT to stream at depth={current_depth}")
                                            else:
                                                log.debug(f"[TTFT] Skipped adding TTFT to stream (already_added={ttft_added_to_stream}, depth={current_depth})")
                                        
                                        # 追蹤每個 content token（用於備用 decode_tp 計算）
                                        if stripped_value and not is_tag_only:
                                            last_content_token_time = current_time
                                            total_output_tokens += 1  # 粗略估計：每個 delta 算 1 個 token

                                        if (
                                            content_blocks
                                            and content_blocks[-1]["type"]
                                            == "reasoning"
                                            and content_blocks[-1]
                                            .get("attributes", {})
                                            .get("type")
                                            == "reasoning_content"
                                        ):
                                            reasoning_block = content_blocks[-1]
                                            reasoning_block["ended_at"] = time.time()
                                            reasoning_block["duration"] = int(
                                                reasoning_block["ended_at"]
                                                - reasoning_block["started_at"]
                                            )

                                            content_blocks.append(
                                                {
                                                    "type": "text",
                                                    "content": "",
                                                }
                                            )

                                        content = f"{content}{value}"
                                        if not content_blocks:
                                            content_blocks.append(
                                                {
                                                    "type": "text",
                                                    "content": "",
                                                }
                                            )

                                        content_blocks[-1]["content"] = (
                                            content_blocks[-1]["content"] + value
                                        )

                                        if DETECT_REASONING:
                                            content, content_blocks, _ = (
                                                tag_content_handler(
                                                    "reasoning",
                                                    reasoning_tags,
                                                    content,
                                                    content_blocks,
                                                )
                                            )

                                        if DETECT_CODE_INTERPRETER:
                                            content, content_blocks, end = (
                                                tag_content_handler(
                                                    "code_interpreter",
                                                    code_interpreter_tags,
                                                    content,
                                                    content_blocks,
                                                )
                                            )

                                            if end:
                                                break

                                        if DETECT_SOLUTION:
                                            content, content_blocks, _ = (
                                                tag_content_handler(
                                                    "solution",
                                                    solution_tags,
                                                    content,
                                                    content_blocks,
                                                )
                                            )

                                        if ENABLE_REALTIME_CHAT_SAVE:
                                            # Save message in the database
                                            Chats.upsert_message_to_chat_by_id_and_message_id(
                                                metadata["chat_id"],
                                                metadata["message_id"],
                                                {
                                                    "content": serialize_content_blocks(
                                                        content_blocks
                                                    ),
                                                },
                                            )
                                        else:
                                            data = {
                                                "content": serialize_content_blocks(
                                                    content_blocks
                                                ),
                                            }

                                if delta:
                                    delta_count += 1
                                    if delta_count >= delta_chunk_size:
                                        await event_emitter(
                                            {
                                                "type": "chat:completion",
                                                "data": data,
                                            }
                                        )
                                        delta_count = 0
                                else:
                                    await event_emitter(
                                        {
                                            "type": "chat:completion",
                                            "data": data,
                                        }
                                    )
                        except Exception as e:
                            done = "data: [DONE]" in line
                            if done:
                                pass
                            else:
                                log.debug(f"Error: {e}")
                                continue

                    if content_blocks:
                        # Clean up the last text block
                        if content_blocks[-1]["type"] == "text":
                            content_blocks[-1]["content"] = content_blocks[-1][
                                "content"
                            ].strip()

                            if not content_blocks[-1]["content"]:
                                content_blocks.pop()

                                if not content_blocks:
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "content": "",
                                        }
                                    )

                        if content_blocks[-1]["type"] == "reasoning":
                            reasoning_block = content_blocks[-1]
                            if reasoning_block.get("ended_at") is None:
                                reasoning_block["ended_at"] = time.time()
                                reasoning_block["duration"] = int(
                                    reasoning_block["ended_at"]
                                    - reasoning_block["started_at"]
                                )

                    if response_tool_calls:
                        tool_calls.append(response_tool_calls)

                    if response.background:
                        await response.background()

                    # 離開函數時減少深度並記錄統計
                    stream_handler_depth -= 1
                    log.debug(f"[TTFT] Stream body handler exited, new depth={stream_handler_depth}")
                    log.info(f"[METRICS] Stream completed - Total chunks: {line_count}, Collected metrics: {metrics_info}, Collected usage: {usage_info}")

                await stream_body_handler(response, form_data)

                MAX_TOOL_CALL_RETRIES = 10
                tool_call_retries = 0

                while len(tool_calls) > 0 and tool_call_retries < MAX_TOOL_CALL_RETRIES:

                    tool_call_retries += 1

                    response_tool_calls = tool_calls.pop(0)

                    content_blocks.append(
                        {
                            "type": "tool_calls",
                            "content": response_tool_calls,
                        }
                    )

                    await event_emitter(
                        {
                            "type": "chat:completion",
                            "data": {
                                "content": serialize_content_blocks(content_blocks),
                            },
                        }
                    )

                    tools = metadata.get("tools", {})

                    results = []

                    for tool_call in response_tool_calls:
                        tool_call_id = tool_call.get("id", "")
                        tool_name = tool_call.get("function", {}).get("name", "")
                        tool_args = tool_call.get("function", {}).get("arguments", "{}")

                        tool_function_params = {}
                        try:
                            # json.loads cannot be used because some models do not produce valid JSON
                            tool_function_params = ast.literal_eval(tool_args)
                        except Exception as e:
                            log.debug(e)
                            # Fallback to JSON parsing
                            try:
                                tool_function_params = json.loads(tool_args)
                            except Exception as e:
                                log.error(
                                    f"Error parsing tool call arguments: {tool_args}"
                                )

                        # Mutate the original tool call response params as they are passed back to the passed
                        # back to the LLM via the content blocks. If they are in a json block and are invalid json,
                        # this can cause downstream LLM integrations to fail (e.g. bedrock gateway) where response
                        # params are not valid json.
                        # Main case so far is no args = "" = invalid json.
                        log.debug(
                            f"Parsed args from {tool_args} to {tool_function_params}"
                        )
                        tool_call.setdefault("function", {})["arguments"] = json.dumps(
                            tool_function_params
                        )

                        tool_result = None

                        if tool_name in tools:
                            tool = tools[tool_name]
                            spec = tool.get("spec", {})

                            try:
                                allowed_params = (
                                    spec.get("parameters", {})
                                    .get("properties", {})
                                    .keys()
                                )

                                tool_function_params = {
                                    k: v
                                    for k, v in tool_function_params.items()
                                    if k in allowed_params
                                }

                                if tool.get("direct", False):
                                    tool_result = await event_caller(
                                        {
                                            "type": "execute:tool",
                                            "data": {
                                                "id": str(uuid4()),
                                                "name": tool_name,
                                                "params": tool_function_params,
                                                "server": tool.get("server", {}),
                                                "session_id": metadata.get(
                                                    "session_id", None
                                                ),
                                            },
                                        }
                                    )

                                else:
                                    tool_function = tool["callable"]
                                    tool_result = await tool_function(
                                        **tool_function_params
                                    )

                            except Exception as e:
                                tool_result = str(e)

                        tool_result_files = []
                        if isinstance(tool_result, list):
                            for item in tool_result:
                                # check if string
                                if isinstance(item, str) and item.startswith("data:"):
                                    tool_result_files.append(item)
                                    tool_result.remove(item)

                        if isinstance(tool_result, dict) or isinstance(
                            tool_result, list
                        ):
                            tool_result = json.dumps(
                                tool_result, indent=2, ensure_ascii=False
                            )

                        results.append(
                            {
                                "tool_call_id": tool_call_id,
                                "content": tool_result,
                                **(
                                    {"files": tool_result_files}
                                    if tool_result_files
                                    else {}
                                ),
                            }
                        )

                    content_blocks[-1]["results"] = results

                    content_blocks.append(
                        {
                            "type": "text",
                            "content": "",
                        }
                    )

                    await event_emitter(
                        {
                            "type": "chat:completion",
                            "data": {
                                "content": serialize_content_blocks(content_blocks),
                            },
                        }
                    )

                    try:
                        # 清理計時資訊後再發送給 LLM
                        cleaned_messages = clean_timing_info_from_messages(form_data["messages"])
                        cleaned_content_blocks_messages = clean_timing_info_from_messages(
                            convert_content_blocks_to_messages(content_blocks, True)
                        )

                        new_form_data = {
                            "model": model_id,
                            "stream": True,
                            "tools": form_data["tools"],
                            "messages": [
                                *cleaned_messages,
                                *cleaned_content_blocks_messages,
                            ],
                        }

                        res = await generate_chat_completion(
                            request,
                            new_form_data,
                            user,
                        )

                        if isinstance(res, StreamingResponse):
                            await stream_body_handler(res, new_form_data)
                        else:
                            break
                    except Exception as e:
                        log.debug(e)
                        break

                if DETECT_CODE_INTERPRETER:
                    MAX_RETRIES = 5
                    retries = 0

                    while (
                        content_blocks[-1]["type"] == "code_interpreter"
                        and retries < MAX_RETRIES
                    ):

                        await event_emitter(
                            {
                                "type": "chat:completion",
                                "data": {
                                    "content": serialize_content_blocks(content_blocks),
                                },
                            }
                        )

                        retries += 1
                        log.debug(f"Attempt count: {retries}")

                        output = ""
                        try:
                            if content_blocks[-1]["attributes"].get("type") == "code":
                                code = content_blocks[-1]["content"]

                                if (
                                    request.app.state.config.CODE_INTERPRETER_ENGINE
                                    == "pyodide"
                                ):
                                    output = await event_caller(
                                        {
                                            "type": "execute:python",
                                            "data": {
                                                "id": str(uuid4()),
                                                "code": code,
                                                "session_id": metadata.get(
                                                    "session_id", None
                                                ),
                                            },
                                        }
                                    )
                                elif (
                                    request.app.state.config.CODE_INTERPRETER_ENGINE
                                    == "jupyter"
                                ):
                                    output = await execute_code_jupyter(
                                        request.app.state.config.CODE_INTERPRETER_JUPYTER_URL,
                                        code,
                                        (
                                            request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_TOKEN
                                            if request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH
                                            == "token"
                                            else None
                                        ),
                                        (
                                            request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD
                                            if request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH
                                            == "password"
                                            else None
                                        ),
                                        request.app.state.config.CODE_INTERPRETER_JUPYTER_TIMEOUT,
                                    )
                                else:
                                    output = {
                                        "stdout": "Code interpreter engine not configured."
                                    }

                                log.debug(f"Code interpreter output: {output}")

                                if isinstance(output, dict):
                                    stdout = output.get("stdout", "")

                                    if isinstance(stdout, str):
                                        stdoutLines = stdout.split("\n")
                                        for idx, line in enumerate(stdoutLines):
                                            if "data:image/png;base64" in line:
                                                image_url = ""
                                                # Extract base64 image data from the line
                                                image_data, content_type = (
                                                    load_b64_image_data(line)
                                                )
                                                if image_data is not None:
                                                    image_url = upload_image(
                                                        request,
                                                        image_data,
                                                        content_type,
                                                        metadata,
                                                        user,
                                                    )
                                                stdoutLines[idx] = (
                                                    f"![Output Image]({image_url})"
                                                )

                                        output["stdout"] = "\n".join(stdoutLines)

                                    result = output.get("result", "")

                                    if isinstance(result, str):
                                        resultLines = result.split("\n")
                                        for idx, line in enumerate(resultLines):
                                            if "data:image/png;base64" in line:
                                                image_url = ""
                                                # Extract base64 image data from the line
                                                image_data, content_type = (
                                                    load_b64_image_data(line)
                                                )
                                                if image_data is not None:
                                                    image_url = upload_image(
                                                        request,
                                                        image_data,
                                                        content_type,
                                                        metadata,
                                                        user,
                                                    )
                                                resultLines[idx] = (
                                                    f"![Output Image]({image_url})"
                                                )
                                        output["result"] = "\n".join(resultLines)
                        except Exception as e:
                            output = str(e)

                        content_blocks[-1]["output"] = output

                        content_blocks.append(
                            {
                                "type": "text",
                                "content": "",
                            }
                        )

                        await event_emitter(
                            {
                                "type": "chat:completion",
                                "data": {
                                    "content": serialize_content_blocks(content_blocks),
                                },
                            }
                        )

                        try:
                            # 清理計時資訊後再發送給 LLM
                            cleaned_messages = clean_timing_info_from_messages(form_data["messages"])
                            cleaned_content = clean_timing_info_from_content(
                                serialize_content_blocks(content_blocks, raw=True)
                            )

                            new_form_data = {
                                "model": model_id,
                                "stream": True,
                                "messages": [
                                    *cleaned_messages,
                                    {
                                        "role": "assistant",
                                        "content": cleaned_content,
                                    },
                                ],
                            }

                            res = await generate_chat_completion(
                                request,
                                new_form_data,
                                user,
                            )

                            if isinstance(res, StreamingResponse):
                                await stream_body_handler(res, new_form_data)
                            else:
                                break
                        except Exception as e:
                            log.debug(e)
                            break

                title = Chats.get_chat_title_by_id(metadata["chat_id"])
                # 計算總處理時間
                total_time = round(time.time() - start_time, 2)

                # 計算 decode throughput
                decode_tp_text = ""
                log.info(f"[METRICS] Final collected data - API metrics: {metrics_info}, API usage: {usage_info}")
                log.info(f"[METRICS] Self-tracked data - first_time: {first_content_token_time}, last_time: {last_content_token_time}, total_tokens: {total_output_tokens}")
                
                # 方案 1: 嘗試使用 API 提供的 metrics
                if metrics_info and usage_info:
                    try:
                        first_token_time = metrics_info.get("first_token_time")
                        last_token_time = metrics_info.get("last_token_time")
                        
                        if first_token_time is not None and last_token_time is not None:
                            decode_time = last_token_time - first_token_time
                            output_tokens_num = usage_info.get("completion_tokens", 0)
                            input_tokens_num = usage_info.get("prompt_tokens", 0)
                            
                            if decode_time > 0 and output_tokens_num > 0:
                                decode_tp = round(output_tokens_num / decode_time, 2)
                                decode_tp_text = f"{decode_tp} tokens/s"
                                log.info(f"[METRICS] ✓ Decode throughput from API - decode_time={round(decode_time, 2)}s, output_tokens={output_tokens_num}, input_tokens={input_tokens_num}, decode_tp={decode_tp} tokens/s")
                            else:
                                log.warning(f"[METRICS] Cannot calculate decode_tp from API - decode_time={decode_time}, output_tokens={output_tokens_num}")
                        else:
                            log.warning(f"[METRICS] Missing token time in API metrics: first_token_time={first_token_time}, last_token_time={last_token_time}")
                    except Exception as e:
                        log.error(f"[METRICS] Error calculating decode throughput from API: {e}")
                
                # 方案 2: 如果 API 沒提供，使用我們自己追蹤的數據（備用方案）
                if not decode_tp_text and first_content_token_time is not None and last_content_token_time is not None:
                    try:
                        decode_time = last_content_token_time - first_content_token_time
                        
                        # 優先使用 API usage，否則使用我們的估計
                        output_tokens_num = usage_info.get("completion_tokens", 0) if usage_info else total_output_tokens
                        
                        if decode_time > 0 and output_tokens_num > 0:
                            decode_tp = round(output_tokens_num / decode_time, 2)
                            decode_tp_text = f"{decode_tp} tokens/s"
                            log.info(f"[METRICS] ✓ Decode throughput from self-tracking (fallback) - decode_time={round(decode_time, 2)}s, output_tokens={output_tokens_num}, decode_tp={decode_tp} tokens/s")
                        else:
                            log.warning(f"[METRICS] Cannot calculate decode_tp from self-tracking - decode_time={decode_time}, output_tokens={output_tokens_num}")
                    except Exception as e:
                        log.error(f"[METRICS] Error calculating decode throughput from self-tracking: {e}")
                
                if not decode_tp_text:
                    log.warning(f"[METRICS] ✗ Could not calculate decode_tp - insufficient data from both API and self-tracking")

                # 記錄最終統計
                log.debug(f"[TTFT] Response completed - TTFT={ttft_value}s, Total Time={total_time}s, stream_handler_depth={stream_handler_depth}")
                # if raw_data_dump:
                #     log.info(f"[TTFT DEBUG] Raw data dump (first 10 lines):")
                #     for dump_line in raw_data_dump:
                #         log.info(f"[TTFT DEBUG]   {dump_line}")

                # 確保時間信息只添加一次
                # 由於已經在串流中添加了 TTFT，這裡只需要確保沒有重複
                log.debug(f"[TTFT] Checking final content: ttft_value={ttft_value}, ttft_added_to_stream={ttft_added_to_stream}, ttft_added_to_final={ttft_added_to_final}")

                # 如果已經在串流中添加過，就不需要再添加到最終內容了
                if ttft_added_to_stream:
                    ttft_added_to_final = True
                    log.debug(f"[TTFT] TTFT was already added to stream, skipping final content addition")
                elif ttft_value > 0 and content_blocks and not ttft_added_to_final:
                    # 只在沒有添加到串流時才添加到最終內容
                    first_text_block = None
                    for block in content_blocks:
                        if block["type"] == "text":
                            first_text_block = block
                            break

                    if first_text_block:
                        already_has_ttft = (
                            first_text_block["content"].startswith("Time to first token:")
                            or first_text_block["content"].startswith("### 🟢Time to first token:")
                        )
                        if not already_has_ttft:
                            first_text_block["content"] = f"### 🟢Time to first token: {ttft_value} s\n{first_text_block['content']}"
                            ttft_added_to_final = True
                        else:
                            ttft_added_to_final = True

                # 只添加一次 Total Time 和 Decode TP
                if not total_time_added:
                    content_blocks[-1]['content'] += f"\nTotal Time: {total_time} s\nToken Rate: {decode_tp_text}"
                    total_time_added = True
                data = {
                    "done": True,
                    "content": serialize_content_blocks(content_blocks),
                    "title": title,
                }

                if not ENABLE_REALTIME_CHAT_SAVE:
                    # Save message in the database
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata["chat_id"],
                        metadata["message_id"],
                        {
                            "content": serialize_content_blocks(content_blocks),
                        },
                    )

                # Send a webhook notification if the user is not active
                if not get_active_status_by_user_id(user.id):
                    webhook_url = Users.get_user_webhook_url_by_id(user.id)
                    if webhook_url:
                        post_webhook(
                            request.app.state.WEBUI_NAME,
                            webhook_url,
                            f"{title} - {request.app.state.config.WEBUI_URL}/c/{metadata['chat_id']}\n\n{content}",
                            {
                                "action": "chat",
                                "message": content,
                                "title": title,
                                "url": f"{request.app.state.config.WEBUI_URL}/c/{metadata['chat_id']}",
                            },
                        )

                await event_emitter(
                    {
                        "type": "chat:completion",
                        "data": data,
                    }
                )

                await background_tasks_handler()
            except asyncio.CancelledError:
                log.warning("Task was cancelled!")
                await event_emitter({"type": "task-cancelled"})

                if not ENABLE_REALTIME_CHAT_SAVE:
                    # Save message in the database
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata["chat_id"],
                        metadata["message_id"],
                        {
                            "content": serialize_content_blocks(content_blocks),
                        },
                    )

            if response.background is not None:
                await response.background()

        # background_tasks.add_task(response_handler, response, events)
        task_id, _ = await create_task(
            request.app.state.redis,
            response_handler(response, events),
            id=metadata["chat_id"],
        )
        return {"status": True, "task_id": task_id}

    else:
        # Fallback to the original response
        async def stream_wrapper(original_generator, events):
            def wrap_item(item):
                return f"data: {item}\n\n"

            for event in events:
                event, _ = await process_filter_functions(
                    request=request,
                    filter_functions=filter_functions,
                    filter_type="stream",
                    form_data=event,
                    extra_params=extra_params,
                )

                if event:
                    yield wrap_item(json.dumps(event))
            async for data in original_generator:
                data, _ = await process_filter_functions(
                    request=request,
                    filter_functions=filter_functions,
                    filter_type="stream",
                    form_data=data,
                    extra_params=extra_params,
                )
                if data:
                    yield data

        return StreamingResponse(
            stream_wrapper(response.body_iterator, events),
            headers=dict(response.headers),
            background=response.background,
        )
