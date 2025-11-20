import logging
import os
from typing import Optional, Union
from pathlib import Path

import requests
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

from urllib.parse import quote
from huggingface_hub import snapshot_download
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from open_webui.config import VECTOR_DB
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT

from open_webui.models.users import UserModel
from open_webui.models.files import Files
from open_webui.models.knowledge import Knowledges
from open_webui.models.notes import Notes

from open_webui.retrieval.vector.main import GetResult
from open_webui.utils.access_control import has_access


from open_webui.env import (
    SRC_LOG_LEVELS,
    OFFLINE_MODE,
    ENABLE_FORWARD_USER_INFO_HEADERS,
)
from open_webui.config import (
    RAG_EMBEDDING_QUERY_PREFIX,
    RAG_EMBEDDING_CONTENT_PREFIX,
    RAG_EMBEDDING_PREFIX_FIELD_NAME,
    RAG_SELF_KM,
    KM_SELF_RAG_API_BASE_URL,
    KM_SELF_RAG_API_FALLBACK,
    KM_RESULT_DIR
)

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


def call_km_rag_api(collection_name: str, question: str, k: int) -> Optional[dict]:
    """
    呼叫 KM RAG API 取得檢索結果

    Args:
        collection_name: 知識庫的 UUID
        question: 使用者問題
        k: 檢索數量

    Returns:
        dict: 轉換後的 query_result 格式，失敗時回傳 None
    """
    try:
        # 準備 API 請求
        api_url = f"{KM_SELF_RAG_API_BASE_URL}/api/v1/query"
        payload = {
            "collection_name": collection_name,
            "question": question,
            "k": k
        }

        log.info(f"Calling KM RAG API: collection={collection_name}, question={question[:50]}...")

        # 發送請求
        response = requests.post(
            api_url,
            json=payload,
            timeout=30,  # 30秒超時
            headers={"Content-Type": "application/json"}
        )

        # 檢查 HTTP 狀態碼
        response.raise_for_status()

        # 解析回應
        result = response.json()
        log.info(f"KM RAG API result: {result}")

        if result.get("success", False):
            filename = result.get("filename", "KM RAG Result")

            raw_path = result.get("file_path", "KM RAG Result").strip()  # 先去除前後空格

            # 如果是相對路徑且以 ./ 開頭，需要判斷
            if raw_path.startswith('./') or raw_path.startswith('.\\'):
                # 如果已經包含完整路徑資訊，可能需要從相對路徑中提取實際路徑
                # 或者根據實際 API 返回的格式處理
                actual_path = Path(raw_path)
            else:
                actual_path = Path(raw_path)

            file_path = Path(KM_RESULT_DIR) / actual_path
            try:
                # 讀取file_path指向的txt檔內容
                with open(file_path, "r", encoding="utf-8") as f:
                    merged_content = f.read()
            except Exception as e:
                log.warning(f"Failed to read file content from {file_path}: {str(e)}")
                merged_content = ""

            log.info(f"KM RAG API success: content_length={len(merged_content)}")

            # 轉換為 query_result 格式
            return {
                "documents": [[merged_content]],  # 雙層列表
                "metadatas": [[{
                    "source": filename,
                    "name": filename,
                    "collection_id": collection_name
                }]]
            }
        else:
            error_msg = result.get("error", "Unknown error")
            log.warning(f"KM RAG API returned error: {error_msg}")
            return None

    except requests.exceptions.RequestException as e:
        log.warning(f"KM RAG API request failed: {str(e)}")
        return None
    except Exception as e:
        log.warning(f"KM RAG API unexpected error: {str(e)}")
        return None


class VectorSearchRetriever(BaseRetriever):
    collection_name: Any
    embedding_function: Any
    top_k: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        result = VECTOR_DB_CLIENT.search(
            collection_name=self.collection_name,
            vectors=[self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)],
            limit=self.top_k,
        )

        ids = result.ids[0]
        metadatas = result.metadatas[0]
        documents = result.documents[0]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results


def query_doc(
    collection_name: str, query_embedding: list[float], k: int, user: UserModel = None
):
    try:
        log.debug(f"query_doc:doc {collection_name}")
        result = VECTOR_DB_CLIENT.search(
            collection_name=collection_name,
            vectors=[query_embedding],
            limit=k,
        )

        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with limit {k}: {e}")
        raise e


def get_doc(collection_name: str, user: UserModel = None):
    try:
        log.debug(f"get_doc:doc {collection_name}")
        result = VECTOR_DB_CLIENT.get(collection_name=collection_name)

        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error getting doc {collection_name}: {e}")
        raise e


def query_doc_with_hybrid_search(
    collection_name: str,
    collection_result: GetResult,
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
) -> dict:
    try:
        log.debug(f"query_doc_with_hybrid_search:doc {collection_name}")
        bm25_retriever = BM25Retriever.from_texts(
            texts=collection_result.documents[0],
            metadatas=collection_result.metadatas[0],
        )
        bm25_retriever.k = k

        vector_search_retriever = VectorSearchRetriever(
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=k,
        )

        if hybrid_bm25_weight <= 0:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_search_retriever], weights=[1.0]
            )
        elif hybrid_bm25_weight >= 1:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever], weights=[1.0]
            )
        else:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_search_retriever],
                weights=[hybrid_bm25_weight, 1.0 - hybrid_bm25_weight],
            )

        compressor = RerankCompressor(
            embedding_function=embedding_function,
            top_n=k_reranker,
            reranking_function=reranking_function,
            r_score=r,
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

        result = compression_retriever.invoke(query)

        distances = [d.metadata.get("score") for d in result]
        documents = [d.page_content for d in result]
        metadatas = [d.metadata for d in result]

        # retrieve only min(k, k_reranker) items, sort and cut by distance if k < k_reranker
        if k < k_reranker:
            sorted_items = sorted(
                zip(distances, metadatas, documents), key=lambda x: x[0], reverse=True
            )
            sorted_items = sorted_items[:k]
            distances, documents, metadatas = map(list, zip(*sorted_items))

        result = {
            "distances": [distances],
            "documents": [documents],
            "metadatas": [metadatas],
        }

        log.info(
            "query_doc_with_hybrid_search:result "
            + f'{result["metadatas"]} {result["distances"]}'
        )
        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with hybrid search: {e}")
        raise e


def merge_get_results(get_results: list[dict]) -> dict:
    # Initialize lists to store combined data
    combined_documents = []
    combined_metadatas = []
    combined_ids = []

    for data in get_results:
        combined_documents.extend(data["documents"][0])
        combined_metadatas.extend(data["metadatas"][0])
        combined_ids.extend(data["ids"][0])

    # Create the output dictionary
    result = {
        "documents": [combined_documents],
        "metadatas": [combined_metadatas],
        "ids": [combined_ids],
    }

    return result


def merge_and_sort_query_results(query_results: list[dict], k: int) -> dict:
    # Initialize lists to store combined data
    combined = dict()  # To store documents with unique document hashes

    for data in query_results:
        distances = data["distances"][0]
        documents = data["documents"][0]
        metadatas = data["metadatas"][0]

        for distance, document, metadata in zip(distances, documents, metadatas):
            if isinstance(document, str):
                doc_hash = hashlib.sha256(
                    document.encode()
                ).hexdigest()  # Compute a hash for uniqueness

                if doc_hash not in combined.keys():
                    combined[doc_hash] = (distance, document, metadata)
                    continue  # if doc is new, no further comparison is needed

                # if doc is alredy in, but new distance is better, update
                if distance > combined[doc_hash][0]:
                    combined[doc_hash] = (distance, document, metadata)

    combined = list(combined.values())
    # Sort the list based on distances
    combined.sort(key=lambda x: x[0], reverse=True)

    # Slice to keep only the top k elements
    sorted_distances, sorted_documents, sorted_metadatas = (
        zip(*combined[:k]) if combined else ([], [], [])
    )

    # Create and return the output dictionary
    return {
        "distances": [list(sorted_distances)],
        "documents": [list(sorted_documents)],
        "metadatas": [list(sorted_metadatas)],
    }


def get_all_items_from_collections(collection_names: list[str]) -> dict:
    results = []

    for collection_name in collection_names:
        if collection_name:
            try:
                result = get_doc(collection_name=collection_name)
                if result is not None:
                    results.append(result.model_dump())
            except Exception as e:
                log.exception(f"Error when querying the collection: {e}")
        else:
            pass

    return merge_get_results(results)


def query_collection(
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
) -> dict:
    results = []
    error = False

    def process_query_collection(collection_name, query_embedding):
        try:
            if collection_name:
                result = query_doc(
                    collection_name=collection_name,
                    k=k,
                    query_embedding=query_embedding,
                )
                if result is not None:
                    return result.model_dump(), None
            return None, None
        except Exception as e:
            log.exception(f"Error when querying the collection: {e}")
            return None, e

    # Generate all query embeddings (in one call)
    query_embeddings = embedding_function(queries, prefix=RAG_EMBEDDING_QUERY_PREFIX)
    log.debug(
        f"query_collection: processing {len(queries)} queries across {len(collection_names)} collections"
    )

    with ThreadPoolExecutor() as executor:
        future_results = []
        for query_embedding in query_embeddings:
            for collection_name in collection_names:
                result = executor.submit(
                    process_query_collection, collection_name, query_embedding
                )
                future_results.append(result)
        task_results = [future.result() for future in future_results]

    for result, err in task_results:
        if err is not None:
            error = True
        elif result is not None:
            results.append(result)

    if error and not results:
        log.warning("All collection queries failed. No results returned.")

    return merge_and_sort_query_results(results, k=k)


def query_collection_with_hybrid_search(
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
    hybrid_bm25_weight: float,
) -> dict:
    results = []
    error = False
    # Fetch collection data once per collection sequentially
    # Avoid fetching the same data multiple times later
    collection_results = {}
    for collection_name in collection_names:
        try:
            log.debug(
                f"query_collection_with_hybrid_search:VECTOR_DB_CLIENT.get:collection {collection_name}"
            )
            collection_results[collection_name] = VECTOR_DB_CLIENT.get(
                collection_name=collection_name
            )
        except Exception as e:
            log.exception(f"Failed to fetch collection {collection_name}: {e}")
            collection_results[collection_name] = None

    log.info(
        f"Starting hybrid search for {len(queries)} queries in {len(collection_names)} collections..."
    )

    def process_query(collection_name, query):
        try:
            result = query_doc_with_hybrid_search(
                collection_name=collection_name,
                collection_result=collection_results[collection_name],
                query=query,
                embedding_function=embedding_function,
                k=k,
                reranking_function=reranking_function,
                k_reranker=k_reranker,
                r=r,
                hybrid_bm25_weight=hybrid_bm25_weight,
            )
            return result, None
        except Exception as e:
            log.exception(f"Error when querying the collection with hybrid_search: {e}")
            return None, e

    # Prepare tasks for all collections and queries
    # Avoid running any tasks for collections that failed to fetch data (have assigned None)
    tasks = [
        (cn, q)
        for cn in collection_names
        if collection_results[cn] is not None
        for q in queries
    ]

    with ThreadPoolExecutor() as executor:
        future_results = [executor.submit(process_query, cn, q) for cn, q in tasks]
        task_results = [future.result() for future in future_results]

    for result, err in task_results:
        if err is not None:
            error = True
        elif result is not None:
            results.append(result)

    if error and not results:
        raise Exception(
            "Hybrid search failed for all collections. Using Non-hybrid search as fallback."
        )

    return merge_and_sort_query_results(results, k=k)


def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    url,
    key,
    embedding_batch_size,
    azure_api_version=None,
):
    if embedding_engine == "":
        return lambda query, prefix=None, user=None: embedding_function.encode(
            query, **({"prompt": prefix} if prefix else {})
        ).tolist()
    elif embedding_engine in ["ollama", "openai", "azure_openai"]:
        func = lambda query, prefix=None, user=None: generate_embeddings(
            engine=embedding_engine,
            model=embedding_model,
            text=query,
            prefix=prefix,
            url=url,
            key=key,
            user=user,
            azure_api_version=azure_api_version,
        )

        def generate_multiple(query, prefix, user, func):
            if isinstance(query, list):
                embeddings = []
                for i in range(0, len(query), embedding_batch_size):
                    embeddings.extend(
                        func(
                            query[i : i + embedding_batch_size],
                            prefix=prefix,
                            user=user,
                        )
                    )
                return embeddings
            else:
                return func(query, prefix, user)

        return lambda query, prefix=None, user=None: generate_multiple(
            query, prefix, user, func
        )
    else:
        raise ValueError(f"Unknown embedding engine: {embedding_engine}")


def get_reranking_function(reranking_engine, reranking_model, reranking_function):
    if reranking_function is None:
        return None
    if reranking_engine == "external":
        return lambda sentences, user=None: reranking_function.predict(
            sentences, user=user
        )
    else:
        return lambda sentences, user=None: reranking_function.predict(sentences)


def get_sources_from_items(
    request,
    items,
    queries,
    embedding_function,
    k,
    reranking_function,
    k_reranker,
    r,
    hybrid_bm25_weight,
    hybrid_search,
    full_context=False,
    user: Optional[UserModel] = None,
):
    """
        從多種項目類型中檢索和提取文檔來源，用於 RAG（檢索增強生成）。

        這個函數處理不同類型的配置（文字、筆記、檔案、集合）並直接檢索相關文檔，
        或對已嵌入的集合執行向量搜索。

        參數說明:
        --------
        request : FastAPI.Request
            包含應用狀態和配置的 FastAPI 請求物件
            用於訪問: request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL

            範例: 來自 FastAPI 請求處理程序的 request 物件

        items : list[dict]
            指定要檢索內容的項目字典列表。每個項目可以是以下類型之一：

            # 1. 純文字項目（臨時上傳的檔案）
            {"type": "text", "id": "123", "name": "doc.txt", "content": "這是一些文字內容"}
            或
            {"type": "text", "file": {"data": {"content": "文字內容", "meta": {"author": "張三"}}}}

            # 2. 筆記項目
            {"type": "note", "id": "note_123"}

            # 3. 檔案項目（完整模式 - 直接返回所有內容）
            {"type": "file", "id": "file_456", "name": "report.pdf", "context": "full"}
            或
            {"type": "file", "id": "file_789", "name": "data.pdf"}  # 使用向量搜索

            # 4. 知識庫/集合項目（完整模式）
            {"type": "collection", "id": "kb_001", "context": "full"}
            或
            {"type": "collection", "id": "kb_002"}  # 使用向量搜索

            # 5. 直接提供的文檔（跳過搜索）
            {"docs": [
                {"content": "文檔1內容", "metadata": {"source": "doc1"}},
                {"content": "文檔2內容", "metadata": {"source": "doc2"}}
            ]}

            # 6. 直接指定集合名稱
            {"collection_name": "file-123"}
            或
            {"collection_names": ["file-123", "file-456"]}

        queries : list[str]
            用於向量相似度搜索的查詢字符串列表
            當項目需要向量搜索時使用（非直接內容檢索）

            範例:
            ["什麼是機器學習？", "解釋神經網路的工作原理"]

        embedding_function : callable
            從文字生成向量的函數，必須接受：
            - query/text: 字符串或字符串列表
            - prefix: 前綴字符串（可選）
            - user: 用戶模型（可選）

            範例函數簽名：
            beauty embedding_function(queries, prefix=None, user=None):
                # 輸入 list[str] 返回 list[list[float]]
                # 輸入 str 返回 list[float]
                return embeddings  # 向量列表

        k : int
            從向量搜索中檢索的頂部文檔數量

            範例: 10  # 檢索前 10 個最相關的文檔

        reranking_function : callable 或 None
            可選的重新排序函數，用於提高搜索結果的相關性
            如果提供，必須接受查詢-文檔對列表並返回分數

            範例:
            beauty reranking_function(query_doc_pairs):
                # query_doc_pairs = [("查詢", "文檔"), ...]
                # 返回分數列表 list[float]
                return [0.95, 0.87, 0.82]

        k_reranker : int
            要重新排序的文檔數量（必須 >= k）

            範例: 20  # 重新排序前 20 個，但只返回 top k 個

        r : float
            重新排序的最小相關性分數閾值（0.0 到 1.0）
            低於此分數的文檔將被過濾掉

            範例: 0.7  # 只保留分數 >= 0.7 的文檔

        hybrid_bm25_weight : float
            混合搜索中 BM25 組件的權重（0.0 到 1.0）
            - 0.0: 純向量搜索
            - 1.0: 純 BM25 搜索
            - 0.5: 兩者權重相等

            範例: 0.3  # 30% BM25 + 70% 向量搜索

        hybrid_search : bool
            是否使用混合搜索（結合 BM25 和向量搜索）
            若為 False，使用純向量搜索

            範例: True

        full_context : bool, 預設 False
            若為 True，從集合中檢索所有文檔而不進行向量搜索
            若為 False，執行向量相似度搜索

            範例: False

        user : UserModel, 可選，預設 None
            用於權限驗證的用戶模型
            用於檢查筆記和集合的訪問權限

        Returns:
        --------
        list[dict]
            源字典列表，每個包含：
            - source: dict - 原始項目字典（如果存在的話會移除 'data' 欄位）
            - document: list[str] - 文檔內容字符串列表
            - metadata: list[dict] - 每個文檔的元數據字典列表
            - distances: list[float] - （可選）相似度/距離分數列表

            範例返回值：
            [
                {
                    "source": {
                        "type": "file",
                        "id": "file_abc123",
                        "name": "報告.pdf"
                    },
                    "document": [
                        "機器學習是人工智慧的一個子集...",
                        "神經網路是受生物神經系統啟發的計算模型..."
                    ],
                    "metadata": [
                        {"file_id": "file_abc123", "name": "報告.pdf", "source": "報告.pdf"},
                        {"file_id": "file_abc123", "name": "報告.pdf", "chunk": 1}
                    ],
                    "distances": [0.95, 0.87]  # 相似度分數，越高越相關
                },
                {
                    "source": {
                        "type": "collection",
                        "id": "knowledge_base_001"
                    },
                    "document": [
                        "深度學習使用多層來學習表示..."
                    ],
                    "metadata": [
                        {"file_id": "doc_xyz", "name": "深度學習.md", "source": "深度學習.md"}
                    ],
                    "distances": [0.82]
                }
            ]

        處理流程:
        ---------
        1. **文字項目**: 直接從項目中提取內容
        - 如果項目有 "file" 數據，使用該數據
        - 否則使用 "content" 欄位

        2. **筆記項目**: 如果用戶有權限，檢索筆記內容
        - 檢查管理員角色、所有權或訪問控制

        3. **檔案項目**:
        - 如果 context="full" 或 BYPASS_EMBEDDING_AND_RETRIEVAL: 直接返回所有檔案內容
        - 否則: 在集合 "file-{id}" 上執行向量搜索

        4. **集合項目**:
        - 如果 context="full" 或 BYPASS_EMBEDDING_AND_RETRIEVAL: 返回知識庫中的所有檔案
        - 否則: 在集合 "{id}" 上執行向量搜索

        5. **文檔項目**: 直接使用提供的文檔和元數據

        6. **集合名稱項目**: 在指定的集合上執行向量搜索

        7. **向量搜索策略**:
        - 如果 full_context=True: 從集合中檢索所有文檔
        - 如果 hybrid_search=True: 使用混合 BM25 + 向量搜索並重新排序
        - 否則: 使用純向量相似度搜索

        8. **重複預防**: 追蹤已提取的集合以避免處理相同的集合兩次
        9. **錯誤處理**: 如果某個項目處理失敗，記錄錯誤並繼續處理其他項目

        Raises:
        -------
        不主動拋出異常，但會記錄錯誤並繼續處理其他項目。

        使用範例:
        --------
        # 範例 1: 檢索已上傳的檔案（完整模式）
        items = [
            {"type": "file", "id": "file_123", "name": "報告.pdf", "context": "full"}
        ]
        sources = get_sources_from_items(
            request=request,
            items=items,
            queries=["機器學習"],
            embedding_function=embed_fn,
            k=5,
            reranking_function=None,
            k_reranker=0,
            r=0.0,
            hybrid_bm25_weight=0.3,
            hybrid_search=False,
            full_context=False,
            user=current_user
        )

        # 範例 2: 從知識庫進行混合搜索
        items = [{"type": "collection", "id": "kb_001"}]
        sources = get_sources_from_items(
            request=request,
            items=items,
            queries=["什麼是深度學習？"],
            embedding_function=embed_fn,
            k=10,
            reranking_function=rerank_fn,
            k_reranker=20,
            r=0.7,
            hybrid_bm25_weight=0.4,
            hybrid_search=True,
            full_context=False,
            user=current_user
        )
    """
    log.debug(
        f"items: {items} {queries} {embedding_function} {reranking_function} {full_context}"
    )

    extracted_collections = []
    query_results = []

    for item in items:
        query_result = None
        collection_names = []

        if item.get("type") == "text":
            # Raw Text
            # Used during temporary chat file uploads

            if item.get("file"):
                # if item has file data, use it
                query_result = {
                    "documents": [
                        [item.get("file", {}).get("data", {}).get("content")]
                    ],
                    "metadatas": [
                        [item.get("file", {}).get("data", {}).get("meta", {})]
                    ],
                }
            else:
                # Fallback to item content
                query_result = {
                    "documents": [[item.get("content")]],
                    "metadatas": [
                        [{"file_id": item.get("id"), "name": item.get("name")}]
                    ],
                }

        elif item.get("type") == "note":
            # Note Attached
            note = Notes.get_note_by_id(item.get("id"))

            if note and (
                user.role == "admin"
                or note.user_id == user.id
                or has_access(user.id, "read", note.access_control)
            ):
                # User has access to the note
                query_result = {
                    "documents": [[note.data.get("content", {}).get("md", "")]],
                    "metadatas": [[{"file_id": note.id, "name": note.title}]],
                }

        elif item.get("type") == "file":
            if (
                item.get("context") == "full"
                or request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
            ):
                if item.get("file", {}).get("data", {}).get("content", ""):
                    # Manual Full Mode Toggle
                    # Used from chat file modal, we can assume that the file content will be available from item.get("file").get("data", {}).get("content")
                    query_result = {
                        "documents": [
                            [item.get("file", {}).get("data", {}).get("content", "")]
                        ],
                        "metadatas": [
                            [
                                {
                                    "file_id": item.get("id"),
                                    "name": item.get("name"),
                                    **item.get("file")
                                    .get("data", {})
                                    .get("metadata", {}),
                                }
                            ]
                        ],
                    }
                elif item.get("id"):
                    file_object = Files.get_file_by_id(item.get("id"))
                    if file_object:
                        query_result = {
                            "documents": [[file_object.data.get("content", "")]],
                            "metadatas": [
                                [
                                    {
                                        "file_id": item.get("id"),
                                        "name": file_object.filename,
                                        "source": file_object.filename,
                                    }
                                ]
                            ],
                        }
            else:
                # Fallback to collection names
                if item.get("legacy"):
                    collection_names.append(f"{item['id']}")
                else:
                    collection_names.append(f"file-{item['id']}")

        elif item.get("type") == "collection":
            if (
                item.get("context") == "full"
                or request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
            ):
                # Manual Full Mode Toggle for Collection
                knowledge_base = Knowledges.get_knowledge_by_id(item.get("id"))

                if knowledge_base and (
                    user.role == "admin"
                    or has_access(user.id, "read", knowledge_base.access_control)
                ):

                    file_ids = knowledge_base.data.get("file_ids", [])

                    documents = []
                    metadatas = []
                    for file_id in file_ids:
                        file_object = Files.get_file_by_id(file_id)

                        if file_object:
                            documents.append(file_object.data.get("content", ""))
                            metadatas.append(
                                {
                                    "file_id": file_id,
                                    "name": file_object.filename,
                                    "source": file_object.filename,
                                }
                            )

                    query_result = {
                        "documents": [documents],
                        "metadatas": [metadatas],
                    }
            else:
                # Check if we should use KM RAG API for collection retrieval
                if RAG_SELF_KM and queries:
                    # Try KM RAG API first
                    api_result = call_km_rag_api(
                        collection_name=item["id"],
                        question=queries[0],  # Use first query (user's last message)
                        k=k
                    )
                    if api_result is not None:
                        # API call successful, use the result
                        query_result = api_result
                        log.info(f"Using KM RAG API result for collection {item['id']}")
                    elif KM_SELF_RAG_API_FALLBACK:
                        # API failed but fallback is enabled, continue with vector search
                        log.info(f"KM RAG API failed, falling back to vector search for collection {item['id']}")
                        if item.get("legacy"):
                            collection_names = item.get("collection_names", [])
                        else:
                            collection_names.append(item["id"])
                    else:
                        # API failed and fallback is disabled, set empty result
                        log.warning(f"KM RAG API failed and fallback disabled for collection {item['id']}")
                        query_result = {
                            "documents": [[]],
                            "metadatas": [[]]
                        }
                else:
                    # Fallback to collection names (original behavior)
                    if item.get("legacy"):
                        collection_names = item.get("collection_names", [])
                    else:
                        collection_names.append(item["id"])

        elif item.get("docs"):
            # BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
            query_result = {
                "documents": [[doc.get("content") for doc in item.get("docs")]],
                "metadatas": [[doc.get("metadata") for doc in item.get("docs")]],
            }
        elif item.get("collection_name"):
            # Direct Collection Name
            collection_names.append(item["collection_name"])
        elif item.get("collection_names"):
            # Collection Names List
            collection_names.extend(item["collection_names"])

        # If query_result is None
        # Fallback to collection names and vector search the collections
        if query_result is None and collection_names:
            collection_names = set(collection_names).difference(extracted_collections)
            if not collection_names:
                log.debug(f"skipping {item} as it has already been extracted")
                continue

            try:
                if full_context:
                    query_result = get_all_items_from_collections(collection_names)
                else:
                    query_result = None  # Initialize to None
                    if hybrid_search:
                        try:
                            query_result = query_collection_with_hybrid_search(
                                collection_names=collection_names,
                                queries=queries,
                                embedding_function=embedding_function,
                                k=k,
                                reranking_function=reranking_function,
                                k_reranker=k_reranker,
                                r=r,
                                hybrid_bm25_weight=hybrid_bm25_weight,
                            )
                        except Exception as e:
                            log.debug(
                                "Error when using hybrid search, using non hybrid search as fallback."
                            )

                    # fallback to non-hybrid search
                    if not hybrid_search and query_result is None:
                        query_result = query_collection(
                            collection_names=collection_names,
                            queries=queries,
                            embedding_function=embedding_function,
                            k=k,
                        )
            except Exception as e:
                log.exception(e)

            extracted_collections.extend(collection_names)

        if query_result:
            if "data" in item:
                del item["data"]
            query_results.append({**query_result, "file": item})

    sources = []
    for query_result in query_results:
        try:
            if "documents" in query_result:
                if "metadatas" in query_result:
                    source = {
                        "source": query_result["file"],
                        "document": query_result["documents"][0],
                        "metadata": query_result["metadatas"][0],
                    }
                    if "distances" in query_result and query_result["distances"]:
                        source["distances"] = query_result["distances"][0]

                    sources.append(source)
        except Exception as e:
            log.exception(e)

    return sources


def get_model_path(model: str, update_model: bool = False):
    # Construct huggingface_hub kwargs with local_files_only to return the snapshot path
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    local_files_only = not update_model

    if OFFLINE_MODE:
        local_files_only = True

    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    # Inspiration from upstream sentence_transformers
    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        # If fully qualified path exists, return input, else set repo_id
        return model
    elif "/" not in model:
        # Set valid repo_id for model short-name
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    # Attempt to query the huggingface_hub library to determine the local path and/or to update
    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model


def generate_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str = "https://api.openai.com/v1",
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_openai_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        r = requests.post(
            f"{url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                **(
                    {
                        "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS and user
                    else {}
                ),
            },
            json=json_data,
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return [elem["embedding"] for elem in data["data"]]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        log.exception(f"Error generating openai batch embeddings: {e}")
        return None


def generate_azure_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    version: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_azure_openai_batch_embeddings:deployment {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        url = f"{url}/openai/deployments/{model}/embeddings?api-version={version}"

        for _ in range(5):
            r = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "api-key": key,
                    **(
                        {
                            "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                            "X-OpenWebUI-User-Id": user.id,
                            "X-OpenWebUI-User-Email": user.email,
                            "X-OpenWebUI-User-Role": user.role,
                        }
                        if ENABLE_FORWARD_USER_INFO_HEADERS and user
                        else {}
                    ),
                },
                json=json_data,
            )
            if r.status_code == 429:
                retry = float(r.headers.get("Retry-After", "1"))
                time.sleep(retry)
                continue
            r.raise_for_status()
            data = r.json()
            if "data" in data:
                return [elem["embedding"] for elem in data["data"]]
            else:
                raise Exception("Something went wrong :/")
        return None
    except Exception as e:
        log.exception(f"Error generating azure openai batch embeddings: {e}")
        return None


def generate_ollama_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    try:
        log.debug(
            f"generate_ollama_batch_embeddings:model {model} batch size: {len(texts)}"
        )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        r = requests.post(
            f"{url}/api/embed",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                **(
                    {
                        "X-OpenWebUI-User-Name": quote(user.name, safe=" "),
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS
                    else {}
                ),
            },
            json=json_data,
        )
        r.raise_for_status()
        data = r.json()

        if "embeddings" in data:
            return data["embeddings"]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        log.exception(f"Error generating ollama batch embeddings: {e}")
        return None


def generate_embeddings(
    engine: str,
    model: str,
    text: Union[str, list[str]],
    prefix: Union[str, None] = None,
    **kwargs,
):
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")
    user = kwargs.get("user")

    if prefix is not None and RAG_EMBEDDING_PREFIX_FIELD_NAME is None:
        if isinstance(text, list):
            text = [f"{prefix}{text_element}" for text_element in text]
        else:
            text = f"{prefix}{text}"

    if engine == "ollama":
        embeddings = generate_ollama_batch_embeddings(
            **{
                "model": model,
                "texts": text if isinstance(text, list) else [text],
                "url": url,
                "key": key,
                "prefix": prefix,
                "user": user,
            }
        )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "openai":
        embeddings = generate_openai_batch_embeddings(
            model, text if isinstance(text, list) else [text], url, key, prefix, user
        )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "azure_openai":
        azure_api_version = kwargs.get("azure_api_version", "")
        embeddings = generate_azure_openai_batch_embeddings(
            model,
            text if isinstance(text, list) else [text],
            url,
            key,
            azure_api_version,
            prefix,
            user,
        )
        return embeddings[0] if isinstance(text, str) else embeddings


import operator
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document


class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        if reranking:
            scores = self.reranking_function(
                [(query, doc.page_content) for doc in documents]
            )
        else:
            from sentence_transformers import util

            query_embedding = self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)
            document_embedding = self.embedding_function(
                [doc.page_content for doc in documents], RAG_EMBEDDING_CONTENT_PREFIX
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        docs_with_scores = list(
            zip(documents, scores.tolist() if not isinstance(scores, list) else scores)
        )
        if self.r_score:
            docs_with_scores = [
                (d, s) for d, s in docs_with_scores if s >= self.r_score
            ]

        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        final_results = []
        for doc, doc_score in result[: self.top_n]:
            metadata = doc.metadata
            metadata["score"] = doc_score
            doc = Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
            final_results.append(doc)
        return final_results
