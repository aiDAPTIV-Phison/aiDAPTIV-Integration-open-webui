"""
RAG Query Service for km-for-agent-builder
整合了 km-for-agent-builder-client 的查詢功能
"""
import os
import json
from typing import Dict, List, Optional
from loguru import logger
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from config import settings

# BM25 imports
try:
    from rank_bm25 import BM25Okapi
    import jieba
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("BM25 dependencies not available. Install with: pip install rank-bm25 jieba")


class RAGQueryService:
    """RAG 查詢服務"""

    def __init__(self):
        self.embedding_model = None
        self.current_model_path = None
        self.collection_cache = {}
        self.bm25_index = None
        self.bm25_documents = []
        # 優先讀取環境變數，如果為空再使用 settings 設定
        env_search_algorithm = os.getenv('SEARCH_ALGORITHM', '').strip()
        if env_search_algorithm:
            self.search_algorithm = env_search_algorithm.lower()
            logger.info(f"Using search algorithm from environment variable: {self.search_algorithm}")
        else:
            self.search_algorithm = settings.SEARCH_ALGORITHM.lower()
            logger.info(f"Using search algorithm from settings: {self.search_algorithm}")

        # 驗證搜尋演算法設定
        if self.search_algorithm not in ['semantic', 'bm25']:
            logger.warning(f"Invalid search algorithm '{self.search_algorithm}', defaulting to 'semantic'")
            self.search_algorithm = 'semantic'

        if self.search_algorithm == 'bm25' and not BM25_AVAILABLE:
            logger.warning("BM25 requested but not available, falling back to semantic search")
            self.search_algorithm = 'semantic'

    def _init_embedding_model(self):
        """初始化嵌入模型"""
        if self.embedding_model is None:
            try:
                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

                # logger.info(f"Loading embedding model from API: {settings.EMBEDDING_URL}")
                # self.embedding_model = HuggingFaceInferenceAPIEmbeddings(
                #     api_url=settings.EMBEDDING_URL,
                #     api_key='empty'
                # )

                from langchain_openai import OpenAIEmbeddings
                # embedding_url = "http://10.101.41.128:13142/v1/"
                logger.info(f"Loading embedding model from API: {settings.EMBEDDING_URL}")
                self.embedding_model = OpenAIEmbeddings(base_url=settings.EMBEDDING_URL, api_key="empty",
                                            tiktoken_enabled=False, check_embedding_ctx_length=False )

                logger.info(f"[SUCCESS] Embedding model loaded from API: {settings.EMBEDDING_URL}")
            except Exception as e:
                logger.exception("[ERROR] Failed to load embedding model from API")
                # 如果 API 載入失敗，回退到本地模型
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    logger.info("Falling back to local HuggingFace model")
                    self.embedding_model = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    logger.info("[SUCCESS] Local embedding model loaded as fallback")
                except Exception as fallback_e:
                    logger.exception("[ERROR] Failed to load fallback embedding model")
                    raise fallback_e

    def _get_collection(self, collection_name: str, chroma_path: str = None):
        """獲取或創建 Chroma collection (帶緩存) - 參考 km-for-agent-builder-client 的實現"""
        if chroma_path is None:
            # 嘗試多個可能的路徑
            possible_paths = [
                settings.CHROMA_PATH,  # 使用配置中的 CHROMA_PATH
                os.path.join(settings.BASE_FOLDER, "chromadb"),
                os.path.join(settings.BASE_FOLDER, collection_name, "processed_output"),
                os.path.join(settings.BASE_FOLDER, collection_name, "processed_output", "chromadb")
            ]

            chroma_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    chroma_path = path
                    break

            if chroma_path is None:
                # 如果都找不到，使用默認路徑
                chroma_path = os.path.join(settings.BASE_FOLDER, collection_name, "processed_output")
                logger.warning(f"No ChromaDB path found, using default: {chroma_path}")

        collection_key = f"{chroma_path}#{collection_name}"

        # 如果是相同的 collection，直接返回緩存的實例
        if collection_key in self.collection_cache:
            logger.debug(f"Using cached collection: {collection_name}")
            return self.collection_cache[collection_key]

        # 創建新的 collection
        logger.info(f"Loading new collection: {collection_name} from path: {chroma_path}")
        if self.embedding_model is None:
            self._init_embedding_model()

        try:
            # 參考 km-for-agent-builder-client 的實現方式
            collection = Chroma(
                persist_directory=chroma_path,
                embedding_function=self.embedding_model,
                collection_name=collection_name
            )
            self.collection_cache[collection_key] = collection
            logger.info(f"Successfully loaded collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Failed to load collection {collection_name} from {chroma_path}: {str(e)}")
            raise e

    def get_available_collections(self) -> List[str]:
        """獲取可用的 collection 列表 - 直接從 ChromaDB 中查找"""
        collections = []

        try:
            # 構建 ChromaDB 的基礎路徑
            chroma_base_path = os.path.join(settings.CHROMA_PATH)

            # 如果 ChromaDB 目錄不存在，嘗試其他可能的路徑
            if not os.path.exists(chroma_base_path):
                # 嘗試在每個 collection 目錄下查找 processed_output/chromadb
                base_folder = settings.BASE_FOLDER
                if os.path.exists(base_folder):
                    for item in os.listdir(base_folder):
                        item_path = os.path.join(base_folder, item)
                        if os.path.isdir(item_path):
                            # 檢查是否有 processed_output 目錄
                            processed_output_path = os.path.join(item_path, "processed_output")
                            if os.path.exists(processed_output_path):
                                # 檢查是否有 chromadb 目錄
                                chroma_path = os.path.join(processed_output_path, "chromadb")
                                if os.path.exists(chroma_path):
                                    chroma_base_path = chroma_path
                                    break

            logger.info(f"Searching for collections in ChromaDB path: {chroma_base_path}")

            if not os.path.exists(chroma_base_path):
                logger.warning(f"ChromaDB path does not exist: {chroma_base_path}")
                return collections

            # 使用 ChromaDB 客戶端直接查詢 collections
            try:
                # 創建 ChromaDB 客戶端
                client = chromadb.PersistentClient(path=chroma_base_path)

                # 獲取所有 collections
                chroma_collections = client.list_collections()

                for collection in chroma_collections:
                    collection_name = collection.name
                    # 檢查 collection 是否有數據
                    try:
                        count = collection.count()
                        if count > 0:
                            collections.append(collection_name)
                            logger.info(f"Found collection '{collection_name}' with {count} documents")
                        else:
                            logger.debug(f"Collection '{collection_name}' is empty, skipping")
                    except Exception as e:
                        logger.warning(f"Error checking collection '{collection_name}': {str(e)}")
                        # 即使無法檢查數量，也嘗試添加（可能 collection 存在但無法訪問）
                        collections.append(collection_name)

                logger.info(f"Found {len(collections)} available collections from ChromaDB: {collections}")

            except Exception as chroma_error:
                logger.error(f"Error accessing ChromaDB: {str(chroma_error)}")
                # 如果 ChromaDB 訪問失敗，回退到文件系統檢查
                logger.info("Falling back to file system check...")
                return self._get_collections_from_filesystem()

        except Exception as e:
            logger.error(f"Error getting available collections: {str(e)}")
            return []

        return collections

    def _get_collections_from_filesystem(self) -> List[str]:
        """從文件系統獲取 collections（備用方法）"""
        collections = []
        base_folder = settings.BASE_FOLDER

        if not os.path.exists(base_folder):
            return collections

        try:
            for item in os.listdir(base_folder):
                item_path = os.path.join(base_folder, item)
                if os.path.isdir(item_path):
                    # 檢查是否有 processed_output 目錄
                    processed_output_path = os.path.join(item_path, "processed_output")
                    if os.path.exists(processed_output_path):
                        # 檢查是否有 chunks.json 文件
                        chunks_file = os.path.join(processed_output_path, "chunks.json")
                        if os.path.exists(chunks_file):
                            collections.append(item)

            logger.info(f"Found {len(collections)} collections from filesystem: {collections}")
            return collections
        except Exception as e:
            logger.error(f"Error getting collections from filesystem: {str(e)}")
            return []

    def clear_collection_cache(self, collection_name: str = None):
        """清除 collection 緩存 - 參考 km-for-agent-builder-client 的實現"""
        if collection_name:
            # 只清除特定 collection 的緩存
            keys_to_remove = [key for key in self.collection_cache.keys() if collection_name in key]
            for key in keys_to_remove:
                del self.collection_cache[key]
            logger.info(f"Cleared collection cache for: {collection_name}")
        else:
            # 清除所有 collection 緩存
            self.collection_cache.clear()
            logger.info("Cleared all collection caches")

    def _tokenize_text(self, text: str) -> List[str]:
        """文本分詞 - 支援中英文"""
        if not BM25_AVAILABLE:
            return text.split()

        # 使用 jieba 進行中文分詞
        tokens = jieba.lcut(text)
        # 過濾掉空白和標點符號
        tokens = [token.strip() for token in tokens if token.strip() and len(token.strip()) > 1]
        return tokens

    def _init_bm25_index(self, collection_name: str):
        """初始化 BM25 索引"""
        if not BM25_AVAILABLE:
            logger.error("BM25 not available")
            return False

        try:
            # 獲取 ChromaDB collection
            chroma = self._get_collection(collection_name)

            # 獲取所有文檔
            all_docs = chroma._collection.get()
            documents = all_docs['documents']
            metadatas = all_docs['metadatas']

            if not documents:
                logger.warning(f"No documents found in collection {collection_name}")
                return False

            # 為每個文檔創建分詞後的文本
            tokenized_docs = []
            self.bm25_documents = []

            for i, doc in enumerate(documents):
                # 結合文檔內容和元數據進行分詞
                full_text = doc
                if metadatas and i < len(metadatas) and metadatas[i]:
                    metadata = metadatas[i]
                    if 'source' in metadata:
                        full_text += f" {metadata['source']}"

                tokenized_doc = self._tokenize_text(full_text)
                tokenized_docs.append(tokenized_doc)
                self.bm25_documents.append({
                    'content': doc,
                    'metadata': metadatas[i] if metadatas and i < len(metadatas) else {},
                    'tokens': tokenized_doc
                })

            # 創建 BM25 索引
            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info(f"BM25 index created for collection {collection_name} with {len(documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {str(e)}")
            return False

    def _bm25_search(self, collection_name: str, question: str, k: int = 5) -> List[Dict]:
        """使用 BM25 進行搜尋"""
        if not BM25_AVAILABLE:
            logger.error("BM25 not available")
            return []

        try:
            # 如果索引不存在，先初始化
            if self.bm25_index is None:
                if not self._init_bm25_index(collection_name):
                    return []

            # 對查詢進行分詞
            query_tokens = self._tokenize_text(question)

            # 使用 BM25 進行搜尋
            scores = self.bm25_index.get_scores(query_tokens)

            # 獲取前 k 個結果
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

            results = []
            for idx in top_indices:
                logger.info(f"BM25 score: {idx} {scores[idx]}")
                if scores[idx] > 0:  # 只返回有分數的結果
                    doc_info = self.bm25_documents[idx]
                    results.append({
                        'content': doc_info['content'],
                        'metadata': doc_info['metadata'],
                        'score': scores[idx]
                    })

            logger.info(f"BM25 search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            return []

    def get_rag_context_with_file_content(self, collection_name: str, question: str, k: int = 5) -> Dict:
        """
        根據問題從 chroma 檢索相關內容，並從對應的 merged file 中讀取完整內容來構建聊天消息

        Args:
            collection_name: 集合名稱
            question: 用戶問題
            k: 檢索的 top-k 數量

        Returns:
            dict: {
                'filename': str,  # 選中的文件名
                'file_path': str,  # 文件的完整路徑
                'chat_messages': List[dict],  # 推理的聊天消息列表
                'merged_content': str,  # 合併的內容
                'error': str  # 錯誤信息，成功時為空字符串
            }
        """
        try:
            # 獲取 collection - 讓 _get_collection 自動尋找正確的路徑
            chroma = self._get_collection(collection_name)

            # 調試信息：檢查 collection 狀態
            try:
                collection_count = chroma._collection.count()
                logger.info(f"Collection '{collection_name}' contains {collection_count} documents")
            except Exception as e:
                logger.warning(f"Could not get collection count: {str(e)}")

            # 根據設定的演算法進行搜尋
            logger.info(f"Searching for: '{question}' with k={k} using {self.search_algorithm} algorithm")

            if self.search_algorithm == 'bm25':
                # 使用 BM25 搜尋
                bm25_results = self._bm25_search(collection_name, question, k)
                if not bm25_results:
                    return {
                        'filename': None,
                        'file_path': None,
                        'chat_messages': [],
                        'merged_content': '',
                        'error': 'no BM25 search results found'
                    }

                # 轉換 BM25 結果格式以匹配語意搜尋的格式
                results = []
                for result in bm25_results:
                    # 創建類似 Document 的對象
                    class MockDocument:
                        def __init__(self, content, metadata):
                            self.page_content = content
                            self.metadata = metadata

                    mock_doc = MockDocument(result['content'], result['metadata'])
                    results.append((mock_doc, result['score']))

            else:
                # 使用語意搜尋（默認）
                results = chroma.similarity_search_with_score(question, k=k)

            logger.info(f"Search returned {len(results)} results")

            if not results:
                return {
                    'filename': None,
                    'file_path': None,
                    'chat_messages': [],
                    'merged_content': '',
                    'error': 'no search results found'
                }

            # 統計每個 group_id 的出現次數和相似度總和
            group_stats = {}
            all_chunks = []

            for doc, score in results:
                group_id = doc.metadata.get('group_id', '')
                chunk_content = doc.page_content
                all_chunks.append(chunk_content)

                if group_id:
                    if group_id not in group_stats:
                        group_stats[group_id] = {
                            'count': 0,
                            'similarity_sum': 0.0,
                            'scores': [],
                            'chunks': []
                        }

                    group_stats[group_id]['count'] += 1
                    group_stats[group_id]['similarity_sum'] += score
                    group_stats[group_id]['scores'].append(score)
                    group_stats[group_id]['chunks'].append(chunk_content)
                    logger.info(f"group_id: {group_id}, similarity_sum: {group_stats[group_id]['similarity_sum']}, scores: {group_stats[group_id]['scores']}")

            logger.info(f"group_stats: {group_stats}")
            if not group_stats:
                return {
                    'filename': None,
                    'file_path': None,
                    'chat_messages': [],
                    'merged_content': '',
                    'error': 'no valid group_ids found'
                }

            # 選擇最相關的 group_id
            max_count = max(stats['count'] for stats in group_stats.values())
            top_groups = [group_id for group_id, stats in group_stats.items()
                         if stats['count'] == max_count]

            if len(top_groups) == 1:
                selected_group_id = top_groups[0]
            else:
                best_group = None
                best_similarity_sum = float('inf')

                for group_id in top_groups:
                    similarity_sum = group_stats[group_id]['similarity_sum']
                    if similarity_sum < best_similarity_sum:
                        best_similarity_sum = similarity_sum
                        best_group = group_id

                selected_group_id = best_group

            logger.info(f"Selected group_id: {selected_group_id}")

            # 構建 merge file 名稱並尋找實際存在的文件
            # 使用 selected_group_id 作為 merged file 的基礎名稱
            group_filename = selected_group_id

            # 構建可能的 merged file 路徑
            base_paths = [
                os.path.join(settings.BASE_FOLDER, collection_name, "merged_files"),
                os.path.join(settings.BASE_FOLDER, collection_name, "processed_output", "merged_files"),
                os.path.join("tmp", collection_name, "processed_output", "merged_files")
            ]

            merged_file_path = None
            merged_file_name = None

            # 嘗試不同的文件名模式
            possible_names = [
                f"{group_filename}.txt",
                f"{group_filename}_merged_part1.txt",
                f"{group_filename}_merged.txt"
            ]

            # 在每個可能的基礎路徑中尋找文件
            for base_path in base_paths:
                if os.path.exists(base_path):
                    for name in possible_names:
                        potential_path = os.path.join(base_path, name)
                        if os.path.exists(potential_path):
                            merged_file_path = potential_path
                            merged_file_name = name
                            logger.info(f"Found merged file: {merged_file_path}")
                            break
                    if merged_file_path:
                        break

            # 如果都沒找到，使用第一個路徑和第一個文件名作為默認
            if merged_file_path is None:
                merged_file_name = possible_names[0]
                merged_file_path = os.path.join(base_paths[0], merged_file_name)
                logger.info(f"Using default merged file path: {merged_file_path}")
            else:
                logger.info(f"Selected merge filename: {merged_file_name}")

            # 從指定的 txt 檔案中讀取內容
            merged_content = ""
            try:
                logger.info(f"Attempting to read merged file: {merged_file_path}")

                if os.path.exists(merged_file_path):
                    with open(merged_file_path, 'r', encoding='utf-8') as f:
                        merged_content = f.read().strip()
                    logger.info(f"Successfully read merged file, content length: {len(merged_content)} chars")
                else:
                    logger.warning(f"Merged file not found: {merged_file_path}")
                    # 如果找不到 merged file，使用檢索到的 chunks
                    merged_content = "\n\n".join(all_chunks)
                    logger.info(f"Using retrieved chunks as fallback, content length: {len(merged_content)} chars")

            except Exception as file_error:
                logger.error(f"Failed to read merged file: {str(file_error)}")
                merged_content = "\n\n".join(all_chunks)
                logger.info(f"Using retrieved chunks as fallback due to error, content length: {len(merged_content)} chars")

            # 創建用於推理的聊天消息
            chat_messages = []

            # 如果 system prompt 不為空，則添加 system 消息
            if settings.SYSTEM_PROMPT and settings.SYSTEM_PROMPT.strip():
                chat_messages.append({
                    "role": "system",
                    "content": settings.SYSTEM_PROMPT
                })

            # 添加 user 消息
            user_prompt_template = settings.USER_PROMPT_TEMPLATE
            user_content = user_prompt_template.format(chunk=merged_content, query=question)
            chat_messages.append({
                "role": "user",
                "content": user_content
            })

            logger.info(f"Suggested merge file name: {merged_file_name if merged_file_name else f'{group_filename}.txt'}")
            logger.info(f"Generated {len(chat_messages)} chat messages")
            logger.debug(f"Retrieved {len(all_chunks)} document chunks")

            return {
                'filename': merged_file_name if merged_file_name else f"{group_filename}.txt",
                'file_path': merged_file_path,
                'chat_messages': chat_messages,
                'merged_content': merged_content,
                'error': ''
            }

        except Exception as e:
            logger.error(f"get_rag_context_with_file_content error: {str(e)}")
            return {
                'filename': None,
                'file_path': None,
                'chat_messages': [],
                'merged_content': '',
                'error': f'internal error: {str(e)}'
            }

    def generate_openai_payload(self, collection_name: str, query: str, k: int = 5,
                               stream: bool = True, model: str = "gpt-4",
                               params: Optional[Dict] = None) -> Dict:
        """
        生成標準 OpenAI 格式的 payload

        Args:
            collection_name: 集合名稱
            query: 用戶問題
            k: 檢索的 top-k 數量
            stream: 是否流式輸出
            model: 模型名稱
            params: 額外參數

        Returns:
            dict: {
                'success': bool,
                'payload_raw': str,
                'message': str
            }
        """
        try:
            # 獲取 RAG 上下文
            result = self.get_rag_context_with_file_content(collection_name, query, k)

            if not result.get("success", True) or result.get("error"):
                return {
                    'success': False,
                    'payload_raw': '',
                    'message': result.get("error", "Failed to get RAG context")
                }

            # 提取文件名
            filename = result.get("filename", "")
            filename_wo_ext = os.path.splitext(filename)[0] if filename else ""

            # 構建標準的 OpenAI 格式 payload
            messages = []

            # 添加 system message
            if settings.SYSTEM_PROMPT and settings.SYSTEM_PROMPT.strip():
                messages.append({
                    "role": "system",
                    "content": settings.SYSTEM_PROMPT
                })

            # 添加用戶消息
            user_content = settings.USER_PROMPT_TEMPLATE.format(
                chunk=result.get("merged_content", ""),
                query=query
            )
            messages.append({
                "role": "user",
                "content": user_content
            })

            # 構建 payload 對象
            payload_obj = {
                "stream": stream,
                "model": model,
                "messages": messages,
                "max_tokens": params.get("max_tokens", 2048) if params else 2048,
                "temperature": params.get("temperature", 0.7) if params else 0.7,
                "top_p": params.get("top_p", 1.0) if params else 1.0,
                "debug_llm_payload": {
                    "km_service_used": True,
                    "collection": collection_name,
                    "filename": filename_wo_ext,
                    "original_query": query,
                    "rag_content_length": len(result.get("merged_content", ""))
                }
            }

            # 轉換為 JSON 字符串
            payload_raw = json.dumps(payload_obj, ensure_ascii=False)

            return {
                'success': True,
                'payload_raw': payload_raw,
                'message': 'OpenAI payload generated successfully'
            }

        except Exception as e:
            logger.error(f"Error generating OpenAI payload: {e}")
            return {
                'success': False,
                'payload_raw': '',
                'message': f"Internal error: {str(e)}"
            }

if __name__ == '__main__':
    # 在測試模式下使用 64 維的假嵌入模型
    class TestFakeEmbeddings:
        def __init__(self, *args, **kwargs):
            pass

        def embed_documents(self, texts):
            # Deterministic pseudo-embeddings based on text length
            import numpy as np
            rng = np.random.default_rng(42)
            vectors = []
            for t in texts:
                length = max(1, len(t))
                rng_local = np.random.default_rng(length)
                vec = rng_local.normal(size=64)  # 64 維，與測試腳本一致
                # L2 normalize
                norm = (vec**2).sum() ** 0.5
                if norm != 0:
                    vec = vec / norm
                vectors.append(vec.tolist())
            return vectors

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    # 創建 RAG 查詢服務並替換嵌入模型
    rag_query_service = RAGQueryService()
    rag_query_service.embedding_model = TestFakeEmbeddings()
    print(f"🧪 測試模式：使用 64 維假嵌入模型，搜尋演算法：{rag_query_service.search_algorithm.upper()}")

    collections = rag_query_service.get_available_collections()
    print(f"Available collections: {collections}")

    # 簡單的 RAG 查詢測試
    if collections:
        test_collection = collections[0]
        print(f"\n測試 RAG 查詢 - Collection: {test_collection}")

        # 先檢查 collection 狀態
        try:
            chroma = rag_query_service._get_collection(test_collection)
            count = chroma._collection.count()
            print(f"Collection 文檔數量: {count}")
        except Exception as e:
            print(f"⚠️  無法獲取 collection 狀態: {str(e)}")

        result = rag_query_service.get_rag_context_with_file_content(
            collection_name=test_collection,
            question="what is NVM ExpressTM",
            k=3
        )

        if result.get('error'):
            print(f"❌ 查詢失敗: {result['error']}")
        else:
            print(f"✅ 查詢成功")
            print(f"   推薦文件: {result.get('filename', 'N/A')}")
            print(f"   消息數量: {len(result.get('chat_messages', []))}")
            # print(result)

        # 測試 generate_openai_payload 功能
        print(f"\n=== 測試 OpenAI Payload 生成 ===")
        try:
            openai_result = rag_query_service.generate_openai_payload(
                collection_name=test_collection,
                query="what is NVM ExpressTM",
                k=3,
                stream=False,
                model="gpt-3.5-turbo",
                params={"temperature": 0.7, "max_tokens": 1000}
            )

            if openai_result['success']:
                print(f"✅ OpenAI Payload 生成成功")
                print(f"   消息: {openai_result['message']}")
                print(f"   Payload 長度: {len(openai_result['payload_raw'])} 字符")

                # 顯示 payload 內容（前 500 字符）
                payload_preview = openai_result['payload_raw'][:500]
                print(f"   Payload 預覽: {payload_preview}...")

                # 嘗試解析 JSON 來驗證格式
                try:
                    import json
                    payload_obj = json.loads(openai_result['payload_raw'])
                    print(f"   ✅ JSON 格式驗證通過")
                    print(f"   模型: {payload_obj.get('model', 'N/A')}")
                    print(f"   流式: {payload_obj.get('stream', 'N/A')}")
                    print(f"   消息數量: {len(payload_obj.get('messages', []))}")
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON 格式錯誤: {str(e)}")
            else:
                print(f"❌ OpenAI Payload 生成失敗: {openai_result['message']}")

        except Exception as e:
            print(f"❌ OpenAI Payload 測試失敗: {str(e)}")
    else:
        print("\n⚠️  沒有可用的 collections")

