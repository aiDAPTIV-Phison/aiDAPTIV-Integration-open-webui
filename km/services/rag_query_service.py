"""
RAG Query Service for km-for-agent-builder
整合了 km-for-agent-builder-client 的查詢功能
"""
import os
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
from loguru import logger
import sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from config import settings, get_user_prompt_template
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from services.bm25_index_manager import BM25IndexManager

# 為了向後相容，檢查 BM25 是否可用（由 BM25IndexManager 處理實際功能）
try:
    from rank_bm25 import BM25Okapi  # noqa: F401
    import jieba  # noqa: F401
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("BM25 dependencies not available. Install with: pip install rank-bm25 jieba")


class RAGQueryService:
    """RAG 查詢服務"""
    
    def __init__(self, embedding_model: Optional[HuggingFaceEmbeddings] = None):
        self.embedding_model = embedding_model
        self.current_model_path = None
        self.bm25_manager = BM25IndexManager()  # BM25 索引管理器
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
        
        if embedding_model == None:
            self.search_algorithm = 'bm25'
    
    def _get_collection(self, collection_name: str, chroma_path: str = None):
        """獲取或創建 Chroma collection - 每次都重新讀取"""

        chroma_path = os.path.join(settings.BASE_FOLDER, collection_name, "chroma_db")
        
        # 每次都重新創建 collection，不使用緩存
        logger.info(f"Loading collection: {collection_name} from path: {chroma_path}")
        
        try:
            collection = Chroma(
                persist_directory=chroma_path,
                embedding_function=self.embedding_model,
                collection_name=collection_name
            )
          
            logger.info(f"Successfully loaded collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Failed to load collection {collection_name} from {chroma_path}: {str(e)}")
            raise e
    
    def clear_bm25_cache(self, collection_name: str = None):
        """清除 BM25 緩存"""
        self.bm25_manager.clear_cache(collection_name)
    
    def get_available_collections(self) -> List[str]:
        """
        獲取可用的 collection 列表
        
        掃描 BASE_FOLDER 下的所有目錄，找出包含 chroma_db 子目錄的目錄作為可用的 collections
        
        Returns:
            可用的 collection 名稱列表
        """
        collections = []
        base_folder = Path(settings.BASE_FOLDER)
        
        if not base_folder.exists():
            logger.warning(f"Base folder does not exist: {base_folder}")
            return collections
        
        # 掃描 base_folder 下的所有目錄
        for item in base_folder.iterdir():
            if item.is_dir():
                # 檢查是否有 chroma_db 子目錄
                chroma_db_path = item / "chroma_db"
                if chroma_db_path.exists() and chroma_db_path.is_dir():
                    collections.append(item.name)
                    logger.debug(f"Found collection: {item.name}")
        
        logger.info(f"Found {len(collections)} available collections: {collections}")
        return sorted(collections)
    
    def get_rag_context_with_file_content(self, chroma: Chroma, collection_name: str, question: str, 
                                          k: int = 5, search_algorithm: Optional[str] = None, language: str = "zh-TW") -> Dict:
        """
        根據問題從 chroma 檢索相關內容，並從對應的 merged file 中讀取完整內容來構建聊天消息
        
        Args:
            collection_name: 集合名稱
            question: 用戶問題
            k: 檢索的 top-k 數量
        
        Returns:
            dict: {
                'filename': str,  # 選中的文件名
                'chat_messages': List[dict],  # 推理的聊天消息列表
                'merged_content': str,  # 合併的內容
                'error': str  # 錯誤信息，成功時為空字符串
            }
        """
        try:
            # 根據設定的演算法進行搜尋（允許覆寫）
            algo = (search_algorithm or self.search_algorithm or 'semantic').lower()
            if algo not in ['semantic', 'bm25']:
                logger.warning(f"Invalid search algorithm '{algo}', defaulting to 'semantic'")
                algo = 'semantic'
            if algo == 'bm25' and not BM25_AVAILABLE:
                logger.warning("BM25 requested but not available, falling back to semantic search")
                algo = 'semantic'
            
            # 如果使用 semantic 但沒有 embedding_model，自動切換到 bm25
            if algo == 'semantic' and self.embedding_model is None:
                logger.warning("Semantic search requested but embedding_model is None, falling back to BM25")
                algo = 'bm25'

            logger.info(f"Searching for: '{question}' with k={k} using {algo} algorithm")
            
            if algo == 'bm25':
                # 使用 BM25 搜尋
                bm25_results = self.bm25_manager.search(chroma, collection_name, question, k, language)
                if not bm25_results:
                    raise ValueError('no BM25 search results found')
                self._show_bm25_results(bm25_results)
                # 轉換 BM25 結果格式以匹配語意搜尋的格式
                # 使用 SimpleNamespace 創建 Document-like 對象來統一介面
                results = []
                for result in bm25_results:
                    # 創建類似 Document 的對象
                    doc = SimpleNamespace(
                        page_content=result['content'],
                        metadata=result['metadata']
                    )
                    # 將 BM25 分數轉換為距離形式（負分數：分數越高，負分數越小，與語意搜尋邏輯一致）
                    # 語意搜尋：分數越低越好（距離越小）
                    # BM25：分數越高越好（相關性越高）
                    # 轉換：使用負分數讓 BM25 分數越高對應距離越小
                    normalized_score = -result['score']
                    results.append((doc, normalized_score))
                
            else:
                # 使用語意搜尋（默認）
                results = chroma.similarity_search_with_score(question, k=k)
            
            logger.info(f"Search returned {len(results)} results")
            
            if not results:
                raise ValueError('no search results found')

            # 顯示搜尋結果
            self._show_search_results(results, max_chunk_length=150)

            # 直接選取第一筆資料的 group_id
            first_doc, first_score = results[0]
            selected_group_id = first_doc.metadata.get('group_id', '')
            if not selected_group_id:
                raise ValueError('no valid group_id found in first result')
            
            # 收集所有 chunks
            all_chunks = [doc.page_content for doc, score in results]
            
            logger.info(f"Selected group_id: {selected_group_id}")
            
            source_filename = selected_group_id
            logger.info(f"Selected source filename: {source_filename}")
            merge_file_name = f"{source_filename}.txt"
            logger.info(f"Selected merge filename: {merge_file_name}")
            
            merged_file_folder = os.path.join(settings.BASE_FOLDER, collection_name, "merged_files")

            # 構建完整的 merged file 路徑
            merged_file_path = os.path.join(
                settings.BASE_FOLDER,
                collection_name,
                "merged_files",
                merge_file_name
            )
            
            # 從指定的 txt 檔案中讀取內容作為 chunk
            merged_content = ""
            try:
                logger.info(f"Attempting to read merged file: {merged_file_path}")
                
                if os.path.exists(merged_file_path):
                    with open(merged_file_path, 'r', encoding='utf-8') as f:
                        merged_content = f.read().strip()
                    logger.info(f"Successfully read merged file, content length: {len(merged_content)} chars")
                else:
                    logger.warning(f"Merged file not found: {merged_file_path}")
                    return {
                        'filename': None,
                        'chat_messages': [],
                        'merged_content': '',
                        'error': 'merge file not found'
                    }
                    
            except Exception as file_error:
                logger.error(f"Failed to read merged file: {str(file_error)}")
                return {
                    'filename': None,
                    'chat_messages': [],
                    'merged_content': '',
                    'error': 'merge file not found'
                }
            
            # 創建用於推理的聊天消息
            chat_messages = []
            
            # 如果 system prompt 不為空，則添加 system 消息
            if settings.SYSTEM_PROMPT and settings.SYSTEM_PROMPT.strip():
                chat_messages.append({
                    "role": "system",
                    "content": settings.SYSTEM_PROMPT
                })
            
            # 添加 user 消息
            chat_messages.append({
                "role": "user", 
                "content": get_user_prompt_template(km_lang=language, include_query=True).format(chunk=merged_content, query=question)
            })
            
            logger.info(f"Suggested merge file name: {merge_file_name}")
            logger.info(f"Generated {len(chat_messages)} chat messages")
            logger.debug(f"Retrieved {len(all_chunks)} document chunks")
            include_file= chroma.get(where={"group_id": selected_group_id})
            # logger.info(f'selected_group_id: {selected_group_id} {include_file}')
            include_file_list = []
            for file in include_file.get('metadatas'):
                source_file = file.get('source_file', '')
                if source_file and source_file not in include_file_list:
                    include_file_list.append(source_file)

            return {
                'filename': merge_file_name,
                'include_file_list': include_file_list,
                'chat_messages': chat_messages,
                'merged_content': merged_content,  # 添加 merged_content 字段
                'retrieved_chunks': all_chunks,  # 新增：檢索到的原始 chunks
                'error': ''
            }
                
        except Exception as e:
            logger.error(f"get_rag_context_with_file_content error: {str(e)}")
            return {
                'filename': None,
                'chat_messages': [],
                'merged_content': '',
                'error': f'internal error: {str(e)}'
            }

    def _show_bm25_results(self, results: List[Dict]):
        simplified_results = []
        for r in results:
            md = r.get("metadata", {}) or {}
            simplified_results.append({
                "score": r.get("score", 0),
                "source_file": md.get("source_file"),
                "group_id": md.get("group_id"),
            })

        output = {
            "count": len(simplified_results),
            "results": simplified_results,
        }
        # 使用 logger 避免 Windows 既定編碼（cp950 等）造成 UnicodeEncodeError
        logger.info(json.dumps(output, ensure_ascii=False, indent=2))
    
    def _show_search_results(self, results: List[tuple], max_chunk_length: int = 100):
        """
        顯示搜尋結果（語意搜尋或 BM25）
        
        Args:
            results: List[Tuple[doc_like, score]]，doc_like 需具有 metadata 與 page_content
            max_chunk_length: chunk 內容最大顯示長度，超過會截斷
        """
        simplified_results = []
        for idx, (doc, score) in enumerate(results):
            chunk_content = doc.page_content
            # 截斷過長的 chunk 內容
            if len(chunk_content) > max_chunk_length:
                chunk_preview = chunk_content[:max_chunk_length] + "..."
            else:
                chunk_preview = chunk_content
            
            md = doc.metadata or {}
            simplified_results.append({
                "index": idx,
                "score": round(score, 4),
                "group_id": md.get("group_id", "N/A"),
                "source_file": md.get("source_file", "N/A"),
                # "chunk_preview": chunk_preview
            })
        
        output = {
            "count": len(simplified_results),
            "results": simplified_results
        }
        # 使用 logger 避免 Windows 既定編碼（cp950 等）造成 UnicodeEncodeError
        logger.info("Search Results:")
        logger.info(json.dumps(output, ensure_ascii=False, indent=2))

    # def _select_group_id(self, results, algo: str):
    #     """
    #     給定檢索結果與演算法，選擇合適的 group_id 並回傳 (selected_group_id, all_chunks)
    #     results: List[Tuple[doc_like, score]]，doc_like 需具有 metadata 與 page_content
    #     """
    #     if not results:
    #         raise ValueError('no search results found')

    #     # BM25：直接選第一筆 group
    #     if algo == 'bm25':
    #         first_group_id = results[0][0].metadata.get('group_id', '')
    #         if not first_group_id:
    #             raise ValueError('no valid group_ids found')
    #         return first_group_id, [results[0][0].page_content]

    #     # Semantic：以次數最多，若並列則取 similarity_sum 較小者
    #     group_stats = {}
    #     all_chunks = []
    #     for doc, score in results:
    #         group_id = doc.metadata.get('group_id', '')
    #         chunk_content = doc.page_content
    #         all_chunks.append(chunk_content)
    #         if group_id:
    #             if group_id not in group_stats:
    #                 group_stats[group_id] = {
    #                     'count': 0,
    #                     'similarity_sum': 0.0,
    #                     'scores': [],
    #                     'chunks': []
    #                 }
    #             group_stats[group_id]['count'] += 1
    #             group_stats[group_id]['similarity_sum'] += score
    #             group_stats[group_id]['scores'].append(score)
    #             group_stats[group_id]['chunks'].append(chunk_content)
    #             logger.info(f"group_id: {group_id}, similarity_sum: {group_stats[group_id]['similarity_sum']}, scores: {group_stats[group_id]['scores']}")

    #     # logger.info(f"group_stats: {group_stats}")
    #     if not group_stats:
    #         raise ValueError('no valid group_ids found')

    #     max_count = max(stats['count'] for stats in group_stats.values())
    #     top_groups = [group_id for group_id, stats in group_stats.items() if stats['count'] == max_count]
    #     if len(top_groups) == 1:
    #         return top_groups[0], all_chunks

    #     best_group = None
    #     best_similarity_sum = float('inf')
    #     for group_id in top_groups:
    #         similarity_sum = group_stats[group_id]['similarity_sum']
    #         if similarity_sum < best_similarity_sum:
    #             best_similarity_sum = similarity_sum
    #             best_group = group_id
    #     return best_group, all_chunks

    def prepare_rag_messages(self, collection_name: str, query: str, k: int = 5, 
                            language: str = "zh-TW") -> Dict:
        """
        準備 RAG 查詢所需的 messages 和上下文資訊（單一職責：只負責 RAG 檢索和 messages 構建）
        
        Args:
            collection_name: 集合名稱
            query: 用戶問題
            k: 檢索的 top-k 數量
            language: 語言設定
        
        Returns:
            dict: {
                'success': bool,
                'messages': List[Dict],  # OpenAI 格式的 messages
                'message': str,
                'merged_file': str,
                'source_files': List[str],
                'retrieved_chunks': List[str],
                'merged_content': str,
                'debug_info': Dict  # 調試資訊
            }
        """
        try:
            # 獲取 collection
            chroma = self._get_collection(collection_name)
            
            # 獲取 RAG 上下文
            result = self.get_rag_context_with_file_content(
                chroma, collection_name, query, k, 
                search_algorithm=self.search_algorithm, 
                language=language
            )
            
            if not result.get("success", True) or result.get("error"):
                return {
                    'success': False,
                    'messages': [],
                    'message': result.get("error", "Failed to get RAG context"),
                    'merged_file': None,
                    'source_files': None,
                    'retrieved_chunks': None,
                    'merged_content': None,
                    'debug_info': {}
                }
            
            # 提取文件資訊
            filename = result.get("filename", "")
            filename_wo_ext = os.path.splitext(filename)[0] if filename else ""
            merged_content = result.get("merged_content", "")
            
            # 構建 messages
            messages = []
            
            # 添加 system message（如果有）
            if settings.SYSTEM_PROMPT and settings.SYSTEM_PROMPT.strip():
                messages.append({
                    "role": "system",
                    "content": settings.SYSTEM_PROMPT
                })
            
            # 添加用戶消息
            user_prompt_template = get_user_prompt_template(km_lang=language, include_query=True)
            user_content = user_prompt_template.format(
                chunk=merged_content,
                query=query
            )
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # 構建調試資訊
            debug_info = {
                "km_service_used": True,
                "collection": collection_name,
                "filename": filename_wo_ext,
                "original_query": query,
                "rag_content_length": len(merged_content),
                "include_file_list": result.get("include_file_list", [])
            }
            
            return {
                'success': True,
                'messages': messages,
                'message': 'RAG messages prepared successfully',
                'merged_file': filename,
                'source_files': result.get("include_file_list", []),
                'retrieved_chunks': result.get("retrieved_chunks", []),
                'merged_content': merged_content,
                'debug_info': debug_info
            }
            
        except Exception as e:
            logger.error(f"Error preparing RAG messages: {e}")
            return {
                'success': False,
                'messages': [],
                'message': f"Internal error: {str(e)}",
                'merged_file': None,
                'source_files': None,
                'retrieved_chunks': None,
                'merged_content': None,
                'debug_info': {}
            }

    def generate_openai_payload(self, collection_name: str, query: str, k: int = 5, 
                               stream: bool = True, model: str = "gpt-4", 
                               params: Optional[Dict] = None, language: str = "zh-TW") -> Dict:
        """
        生成標準 OpenAI 格式的 payload（單一職責：只負責組裝和序列化 payload）
        
        Args:
            collection_name: 集合名稱
            query: 用戶問題
            k: 檢索的 top-k 數量
            stream: 是否流式輸出
            model: 模型名稱
            params: 額外參數
            language: 語言設定
        
        Returns:
            dict: {
                'success': bool,
                'payload_raw': str,  # JSON 格式的 payload 字符串
                'message': str,
                'merged_file': str,
                'source_files': List[str],
                'retrieved_chunks': List[str],
                'merged_content': str
            }
        """
        try:
            # 使用新的 prepare_rag_messages 函數
            rag_result = self.prepare_rag_messages(collection_name, query, k, language)
            
            if not rag_result['success']:
                return {
                    'success': False,
                    'payload_raw': '',
                    'message': rag_result['message'],
                    'merged_file': None,
                    'source_files': None,
                    'retrieved_chunks': None,
                    'merged_content': None
                }
            
            # 構建完整的 payload 對象
            payload_obj = {
                "stream": stream,
                "model": model,
                "messages": rag_result['messages'],
                "max_tokens": params.get("max_tokens", 2048) if params else 2048,
                "temperature": params.get("temperature", 0.7) if params else 0.7,
                "top_p": params.get("top_p", 1.0) if params else 1.0,
                # "debug_llm_payload": rag_result['debug_info']
            }
            
            # 序列化為 JSON 字符串
            payload_raw = json.dumps(payload_obj, ensure_ascii=False)
            
            return {
                'success': True,
                'payload_raw': payload_raw,
                'message': 'OpenAI payload generated successfully',
                'merged_file': rag_result['merged_file'],
                'source_files': rag_result['source_files'],
                'retrieved_chunks': rag_result['retrieved_chunks'],
                'merged_content': rag_result['merged_content']
            }
            
        except Exception as e:
            logger.error(f"Error generating OpenAI payload: {e}")
            return {
                'success': False,
                'payload_raw': '',
                'message': f"Internal error: {str(e)}",
                'merged_file': None,
                'source_files': None,
                'retrieved_chunks': None,
                'merged_content': None
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
    logger.info(f"🧪 測試模式：使用 64 維假嵌入模型，搜尋演算法：{rag_query_service.search_algorithm.upper()}")

    collections = rag_query_service.get_available_collections()
    logger.info(f"Available collections: {collections}")
    
    # 簡單的 RAG 查詢測試
    if collections:
        test_collection = collections[0]
        logger.info(f"\n測試 RAG 查詢 - Collection: {test_collection}")
        
        # # 先檢查 collection 狀態
        # try:
        #     chroma = rag_query_service._get_collection(test_collection)
        #     count = chroma._collection.count()
        #     logger.info(f"Collection 文檔數量: {count}")
        # except Exception as e:
        #     logger.info(f"⚠️  無法獲取 collection 狀態: {str(e)}")
        
        result = rag_query_service.get_rag_context_with_file_content(
            collection_name=test_collection,
            question="what is NVM ExpressTM",
            k=3
        )
        
        if result.get('error'):
            logger.info(f"❌ 查詢失敗: {result['error']}")
        else:
            logger.info(f"✅ 查詢成功")
            logger.info(f"   推薦文件: {result.get('filename', 'N/A')}")
            logger.info(f"   消息數量: {len(result.get('chat_messages', []))}")
            # logger.info(result)
        
        # 測試 generate_openai_payload 功能
        logger.info(f"\n=== 測試 OpenAI Payload 生成 ===")
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
                logger.info(f"✅ OpenAI Payload 生成成功")
                logger.info(f"   消息: {openai_result['message']}")
                logger.info(f"   Payload 長度: {len(openai_result['payload_raw'])} 字符")
                
                # 顯示 payload 內容（前 500 字符）
                payload_preview = openai_result['payload_raw'][:500]
                logger.info(f"   Payload 預覽: {payload_preview}...")
                
                # 嘗試解析 JSON 來驗證格式
                try:
                    import json
                    payload_obj = json.loads(openai_result['payload_raw'])
                    logger.info(f"   ✅ JSON 格式驗證通過")
                    logger.info(f"   模型: {payload_obj.get('model', 'N/A')}")
                    logger.info(f"   流式: {payload_obj.get('stream', 'N/A')}")
                    logger.info(f"   消息數量: {len(payload_obj.get('messages', []))}")
                except json.JSONDecodeError as e:
                    logger.info(f"   ❌ JSON 格式錯誤: {str(e)}")
            else:
                logger.info(f"❌ OpenAI Payload 生成失敗: {openai_result['message']}")
                
        except Exception as e:
            logger.info(f"❌ OpenAI Payload 測試失敗: {str(e)}")
    else:
        logger.info("\n⚠️  沒有可用的 collections")

