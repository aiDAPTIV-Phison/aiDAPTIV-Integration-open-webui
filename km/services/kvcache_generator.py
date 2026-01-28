"""
KV Cache generation service (lightweight)
Only performs per-file processing by calling external LLM API.
No llama.cpp (llama-server) start/stop or health checks here.
"""

import os
from typing import List, Dict, Any
from loguru import logger
from langchain_community.docstore.document import Document
import httpx

from config import settings, get_user_prompt_template


class KVCacheGeneratorService:
    """Lightweight KV Cache generator without managing llama.cpp lifecycle"""

    def __init__(self, collection_name: str = "default", base_folder: str = "./data"):
        self.llm_api_url = settings.LLM_API_URL
        self.llm_api_key = settings.LLM_API_KEY
        self.collection_name = collection_name
        self.base_folder = base_folder

        # Where llama-server (managed elsewhere) will save KV cache files
        self.collection_folder = os.path.join(self.base_folder, self.collection_name)

        # HTTP client
        self.client = httpx.AsyncClient()

        # Request behavior
        self.request_timeout = 3600

        logger.info(f"KVCacheGeneratorService(lightweight) initialized: API={self.llm_api_url}")
        logger.info(f"KV cache path (slot-save-path expected): {self.collection_folder}")

    async def generate_kvcaches_for_collection(self, collection_name: str, merged_files: List[str], language: str = "zh-TW") -> int:
        """
        Generate KV cache for all merged files of a collection by calling LLM API per file.
        This implementation DOES NOT start/stop llama.cpp; assumes service is already up.

        Args:
            collection_name: Collection name
            merged_files: List of merged file paths
            language: Language for prompt template (zh-TW, en, english)

        Returns:
            Number of files processed successfully
        """
        try:
            logger.info(f"Start (lightweight) KV cache generation for collection '{collection_name}', files: {len(merged_files)}")

            processed_count = 0

            for i, file_path in enumerate(merged_files):
                logger.info(f"Processing {i+1}/{len(merged_files)}: {os.path.basename(file_path)}")

                try:
                    merge_content = self._load_merged_file(file_path)

                    full_filename = os.path.basename(file_path)
                    if full_filename.endswith('.txt'):
                        original_filename = full_filename[:-4]
                    else:
                        original_filename = full_filename

                    task_id = f"{collection_name}_{original_filename}"

                    await self._generate_single_kvcache(merge_content, task_id, language)
                    processed_count += 1

                    logger.info(f"\u2713 KV cache generated for '{original_filename}'")

                except Exception as e:
                    logger.error(f"\u2717 Failed to process file {file_path}: {str(e)}")
                    continue

            logger.info(f"Collection '{collection_name}' KV cache done: {processed_count}/{len(merged_files)} succeeded")
            return processed_count

        except Exception as e:
            logger.error(f"Collection '{collection_name}' KV cache generation failed (lightweight): {str(e)}")
            raise

    def _load_merged_file(self, file_path: str) -> List[Document]:
        """
        Load content from merged file as Document list.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            document = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "file_name": os.path.basename(file_path)
                }
            )

            return document

        except Exception as e:
            logger.error(f"Failed to load merged file: {file_path}, error: {str(e)}")
            raise

    async def _generate_single_kvcache(self, merge_content: Document, task_id: str, language: str = "zh-TW") -> None:
        """
        Generate KV cache for single merged content by calling LLM API.
        Assumes llama.cpp service is already running elsewhere.
        """
        try:
            await self._call_llm_api(merge_content, task_id, language)
        except Exception as e:
            logger.error(f"Failed to generate single KV cache: task_id={task_id}, error: {str(e)}")
            raise

    async def _call_llm_api(self, document_data: Document, task_id: str, language: str = "zh-TW") -> None:
        """
        Call external LLM API to generate KV cache.
        Relies on server-side `cache_prompt` to persist KV automatically.
        """
        try:
            logger.info(f"Start generating KV cache (lightweight) for task {task_id}")

            merged_content = document_data.page_content

            user_prompt_template = get_user_prompt_template(km_lang=language, include_query=False)
            user_content = user_prompt_template.format(chunk=merged_content)

            messages = []
            if settings.SYSTEM_PROMPT and settings.SYSTEM_PROMPT.strip():
                messages.append({
                    "role": "system",
                    "content": settings.SYSTEM_PROMPT
                })

            messages.append({
                "role": "system",
                "content": user_content
            })

            # logger.info(f"{messages=}")

            request_data = {
                "messages": messages,
                "model": settings.LLM_MODEL_NAME,
                "max_tokens": 2,
                "temperature": 0,
                "stream": False,
                "cache_prompt": True,
                "offload_folder_name": self.collection_name
            }
            logger.info(f"{request_data}")

            headers = {
                "Content-Type": "application/json"
            }
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"

            logger.info(f"Sending request to LLM API (lightweight), content length: {len(merged_content)}")

            response = await self.client.post(
                self.llm_api_url,
                json=request_data,
                headers=headers,
                timeout=self.request_timeout
            )

            response.raise_for_status()
            result = response.json()

            logger.info(f"LLM API response successful: {result.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")
            logger.info(f"KV cache expected to be saved by server to: {self.collection_folder}")

        except Exception as e:
            logger.error(f"LLM API call failed (lightweight): {str(e)}")
            raise

    async def close(self):
        await self.client.aclose()
        logger.info("KVCacheGeneratorService(lightweight) resources closed")


