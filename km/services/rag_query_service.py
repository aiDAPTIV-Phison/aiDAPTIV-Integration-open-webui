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
                    logger.info(f"Found collection: {item.name}")
        
        logger.info(f"Found {len(collections)} available collections: {collections}")
        return sorted(collections)
    
    def _execute_search(self, chroma: Chroma, collection_name: str, question: str, 
                         k: int, search_algorithm: Optional[str], language: str,
                         log_prefix: str = "") -> List[tuple]:
        """
        執行搜尋邏輯（共用方法）
        
        Args:
            chroma: Chroma collection
            collection_name: 集合名稱
            question: 用戶問題
            k: 檢索的 top-k 數量
            search_algorithm: 搜尋演算法 ('semantic' 或 'bm25')
            language: 語言設定
            log_prefix: 日誌前綴標識
        
        Returns:
            List[tuple]: 搜尋結果列表 [(doc, score), ...]
        
        Raises:
            ValueError: 搜尋無結果時拋出
        """
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

        logger.info(f"{log_prefix}Searching for: '{question}' with k={k} using {algo} algorithm")
        
        if algo == 'bm25':
            # 使用 BM25 搜尋
            bm25_results = self.bm25_manager.search(chroma, collection_name, question, k, language)
            if not bm25_results:
                raise ValueError('no BM25 search results found')
            self._show_bm25_results(bm25_results)
            # 轉換 BM25 結果格式以匹配語意搜尋的格式
            results = []
            for result in bm25_results:
                doc = SimpleNamespace(
                    page_content=result['content'],
                    metadata=result['metadata']
                )
                # 將 BM25 分數轉換為距離形式（負分數：分數越高，負分數越小，與語意搜尋邏輯一致）
                normalized_score = -result['score']
                results.append((doc, normalized_score))
        else:
            # 使用語意搜尋（默認）
            results = chroma.similarity_search_with_score(question, k=k)
        
        logger.info(f"{log_prefix}Search returned {len(results)} results")
        
        if not results:
            raise ValueError('no search results found')

        # 顯示搜尋結果
        self._show_search_results(results, max_chunk_length=150)
        
        return results

    def _build_chat_messages(self, merged_content: str, question: str, language: str) -> List[Dict]:
        """
        創建用於推理的聊天消息（共用方法）
        
        Args:
            merged_content: 合併的內容
            question: 用戶問題
            language: 語言設定
        
        Returns:
            List[Dict]: 聊天消息列表
        """
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
        
        return chat_messages

    def _build_error_response(self, error_msg: str) -> Dict:
        """
        建立錯誤回應（共用方法）
        
        Args:
            error_msg: 錯誤信息
        
        Returns:
            Dict: 錯誤回應字典
        """
        return {
            'filename': None,
            'include_file_list': [],
            'chat_messages': [],
            'merged_content': '',
            'retrieved_chunks': [],
            'error': error_msg
        }

    def get_rag_context_with_file_content(self, chroma: Chroma, collection_name: str, question: str, 
                                          k: int = 5, search_algorithm: Optional[str] = None, language: str = "zh-TW") -> Dict:
        """
        根據問題從 chroma 檢索相關內容，並從對應的 merged file 中讀取完整內容來構建聊天消息
        
        Args:
            chroma: Chroma collection
            collection_name: 集合名稱
            question: 用戶問題
            k: 檢索的 top-k 數量
            search_algorithm: 搜尋演算法 ('semantic' 或 'bm25')
            language: 語言設定
        
        Returns:
            dict: {
                'filename': str,  # 選中的文件名
                'include_file_list': List[str],  # 來源文件列表
                'chat_messages': List[dict],  # 推理的聊天消息列表
                'merged_content': str,  # 合併的內容
                'retrieved_chunks': List[str],  # 檢索到的原始 chunks
                'error': str  # 錯誤信息，成功時為空字符串
            }
        """
        try:
            # 執行搜尋
            results = self._execute_search(chroma, collection_name, question, k, search_algorithm, language)

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

            # 構建完整的 merged file 路徑
            merged_file_path = os.path.join(
                settings.BASE_FOLDER,
                collection_name,
                "merged_files",
                merge_file_name
            )
            
            # 從指定的 txt 檔案中讀取內容作為 chunk
            try:
                logger.info(f"Attempting to read merged file: {merged_file_path}")
                
                if os.path.exists(merged_file_path):
                    with open(merged_file_path, 'r', encoding='utf-8') as f:
                        merged_content = f.read().strip()
                    logger.info(f"Successfully read merged file, content length: {len(merged_content)} chars")
                else:
                    logger.warning(f"Merged file not found: {merged_file_path}")
                    return self._build_error_response('merge file not found')
                    
            except Exception as file_error:
                logger.error(f"Failed to read merged file: {str(file_error)}")
                return self._build_error_response('merge file not found')
            
            # 創建聊天消息
            chat_messages = self._build_chat_messages(merged_content, question, language)
            
            logger.info(f"Suggested merge file name: {merge_file_name}")
            logger.info(f"Generated {len(chat_messages)} chat messages")
            logger.info(f"Retrieved {len(all_chunks)} document chunks")
            
            # 收集來源文件列表
            include_file = chroma.get(where={"group_id": selected_group_id})
            include_file_list = []
            for file in include_file.get('metadatas'):
                source_file = file.get('source_file', '')
                if source_file and source_file not in include_file_list:
                    include_file_list.append(source_file)

            return {
                'filename': merge_file_name,
                'include_file_list': include_file_list,
                'chat_messages': chat_messages,
                'merged_content': merged_content,
                'retrieved_chunks': all_chunks,
                'error': ''
            }
                
        except Exception as e:
            logger.error(f"get_rag_context_with_file_content error: {str(e)}")
            return self._build_error_response(f'internal error: {str(e)}')

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
                "chunk_id": md.get("chunk_id", "N/A"),
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

    def _expand_and_deduplicate_parent_chunks(
        self, 
        results: List[tuple], 
        chroma: Optional[Chroma] = None
    ) -> List[tuple]:
        """
        擴展為 parent chunks 並使用區間合併算法進行去重
        
        實現 Small-to-Big 檢索策略（優化版）：
        1. 按來源文件分組
        2. 對同一文件內的 chunk 索引進行區間合併，避免重疊
        3. 合併後動態組裝 parent chunk 內容
        4. 不同文件的結果直接保留
        
        Args:
            results: List[(Document, score)] - 檢索結果
            chroma: Chroma 向量數據庫實例（用於動態獲取 chunk 內容）
        
        Returns:
            去重後的結果列表，page_content 已替換為動態組裝的 parent_chunk_content
        """
        if not results:
            logger.info("No results to expand and deduplicate")
            return results
        
        logger.info(f"Starting interval merge deduplication for {len(results)} search results...")
        
        # Step 1: 按來源文件分組
        from collections import defaultdict
        results_by_source = defaultdict(list)
        
        for idx, (doc, score) in enumerate(results):
            source = doc.metadata.get('source', 'unknown')
            chunk_index = doc.metadata.get('chunk_index', 0)
            chunk_id = doc.metadata.get('chunk_id', f'unknown_{idx}')
            total_chunks = doc.metadata.get('total_chunks', 0)
            parent_chunk_ids_json = doc.metadata.get('parent_chunk_ids', '')
            
            # 解析 parent_chunk_ids
            parent_chunk_ids = []
            if parent_chunk_ids_json:
                try:
                    parent_chunk_ids = json.loads(parent_chunk_ids_json)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse parent_chunk_ids for chunk: {chunk_id}")
            
            results_by_source[source].append({
                'doc': doc,
                'score': score,
                'chunk_index': chunk_index,
                'chunk_id': chunk_id,
                'parent_chunk_ids': parent_chunk_ids,
                'total_chunks': total_chunks
            })
        
        logger.info(f"Grouped by file: {len(results_by_source)} files")
        
        # Step 2: 如果有 Chroma 實例，預先獲取所有需要的 chunk 內容
        chunk_content_cache = {}
        if chroma is not None:
            chunk_content_cache = self._build_chunk_content_cache(chroma, results_by_source)
        
        # Step 3: 對每個文件內的結果進行區間合併
        deduplicated_results = []
        total_merged = 0
        
        for source, file_results in results_by_source.items():
            # 按 chunk_index 排序
            file_results.sort(key=lambda x: x['chunk_index'])
            
            # 獲取該文件的總 chunk 數和最小 chunk_index（用於邊界檢查）
            total_chunks = file_results[0].get('total_chunks', 0)
            min_chunk_index = min(r['chunk_index'] for r in file_results)
            
            # 計算每個 chunk 的 parent 區間 (window_size = 1)
            # parent 區間 = [chunk_index-1, chunk_index+1]，但不能小於最小 chunk_index
            intervals = []
            for r in file_results:
                idx = r['chunk_index']
                # parent 區間: [前一個, 當前, 後一個]
                start = max(min_chunk_index, idx - 1)  # 使用實際的最小 chunk_index
                end = min(total_chunks, idx + 1) if total_chunks > 0 else idx + 1
                intervals.append({
                    'start': start,
                    'end': end,
                    'data': r
                })
            
            # 區間合併算法
            merged_intervals = self._merge_intervals(intervals)
            
            merged_count = len(file_results) - len(merged_intervals)
            if merged_count > 0:
                total_merged += merged_count
                logger.info(
                    f"File {source}: {len(file_results)} results merged into {len(merged_intervals)} intervals "
                    f"(merged {merged_count} overlapping intervals)"
                )
            
            # 為每個合併後的區間動態組裝 parent chunk
            for interval in merged_intervals:
                best_result = interval['best_data']
                
                # 收集此區間內所有 result 的 parent_chunk_ids
                all_parent_chunk_ids = []
                hit_chunk_ids = []  # 真正被向量命中的 chunk_ids
                
                for r in file_results:
                    if interval['start'] <= r['chunk_index'] <= interval['end']:
                        parent_chunk_ids = r.get('parent_chunk_ids', [])
                        all_parent_chunk_ids.extend(parent_chunk_ids)
                        # 找出真正命中的 chunk_id（通常是 parent_chunk_ids 的中間元素）
                        hit_chunk_ids.append(r['chunk_id'])
                
                # 去重並保持順序（假設 chunk_ids 已按順序排列）
                seen = set()
                merged_chunk_ids = []
                for chunk_id in all_parent_chunk_ids:
                    if chunk_id not in seen:
                        seen.add(chunk_id)
                        merged_chunk_ids.append(chunk_id)
                
                logger.info(
                    f"Processing merged interval [{interval['start']}, {interval['end']}]: "
                    f"contains {len(hit_chunk_ids)} hit points, "
                    f"merged {interval['merged_count']} original search results, "
                    f"total {len(merged_chunk_ids)} chunks"
                )
                
                # 統一使用 parent_chunk_ids 組裝
                if merged_chunk_ids and chunk_content_cache:
                    parent_content = self._assemble_parent_chunk_by_ids(
                        chunk_ids=merged_chunk_ids,
                        current_chunk_id=hit_chunk_ids,  # 傳入多個命中點
                        chunk_cache=chunk_content_cache
                    )
                    logger.info(
                        f"Dynamically assembled parent chunk: {len(merged_chunk_ids)} chunks, "
                        f"{len(hit_chunk_ids)} hit points"
                    )
                else:
                    # 無 parent_chunk_ids 或 cache，使用原始內容
                    parent_content = best_result['doc'].page_content
                    logger.warning(
                        f"Cannot dynamically assemble {best_result['chunk_id']}, using original content"
                    )
                
                # 擴展 metadata，添加區間和命中信息
                expanded_metadata = best_result['doc'].metadata.copy()
                expanded_metadata['merged_interval'] = (interval['start'], interval['end'])
                expanded_metadata['hit_chunk_ids'] = hit_chunk_ids
                
                expanded_doc = SimpleNamespace(
                    page_content=parent_content,
                    metadata=expanded_metadata
                )
                deduplicated_results.append((expanded_doc, best_result['score']))
        
        # 按分數重新排序（因為合併可能打亂順序）
        deduplicated_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(
            f"Interval merge deduplication completed: {len(results)} -> {len(deduplicated_results)} chunks "
            f"(merged {total_merged} overlapping intervals)"
        )
        
        return deduplicated_results
    
    def _build_xml_merged_content(
        self, 
        expanded_results: List[tuple],
        chunk_content_cache: dict
    ) -> str:
        """
        將擴展後的檢索結果構建為 XML 格式的 merged_content
        
        按 source 分組，每個文件生成一個 <document> 標籤
        在文件內標記原始命中的區域為 [Focus Area Start/End]
        
        Args:
            expanded_results: List[(expanded_doc, score)] - 擴展後的檢索結果
            chunk_content_cache: dict - chunk 內容快取 {chunk_id: content}
        
        Returns:
            XML 格式的 merged_content 字符串
        """
        from collections import defaultdict
        
        # 按 source 分組
        results_by_source = defaultdict(list)
        for doc, score in expanded_results:
            source = doc.metadata.get('source', 'unknown')
            results_by_source[source].append((doc, score))
        
        # 為每個文件生成 XML document
        xml_documents = []
        for doc_idx, (source, docs) in enumerate(results_by_source.items(), 1):
            filename = source.split('/')[-1] if '/' in source else source.split('\\')[-1] if '\\' in source else source
            
            # 收集該文件的所有區間
            segments = []
            for doc, score in docs:
                merged_interval = doc.metadata.get('merged_interval', None)
                hit_indices = doc.metadata.get('hit_indices', [])
                
                if merged_interval:
                    start_idx, end_idx = merged_interval
                    hit_set = set(hit_indices)
                    
                    # 構建該區間的內容，標記 Focus Area
                    segment_parts = []
                    is_in_focus = False
                    
                    for i in range(start_idx, end_idx + 1):
                        # 從 cache 獲取 chunk 內容
                        chunk_text = ""
                        if source in chunk_content_cache and i in chunk_content_cache[source]:
                            chunk_text = chunk_content_cache[source][i]
                        elif not chunk_text:
                            # 嘗試使用 chunk_id 格式
                            chunk_id = f"{source}#chunk_{i}"
                            chunk_text = chunk_content_cache.get(chunk_id, "")
                        
                        if not chunk_text:
                            # 嘗試從 page_content 中提取（fallback）
                            if i == start_idx and not hit_set:
                                chunk_text = doc.page_content
                        
                        chunk_text = chunk_text.strip()
                        if not chunk_text:
                            continue
                        
                        # 處理 Focus Area 標籤
                        if i in hit_set and not is_in_focus:
                            segment_parts.append("[Focus Area Start]")
                            is_in_focus = True
                        elif i not in hit_set and is_in_focus:
                            segment_parts.append("[Focus Area End]")
                            is_in_focus = False
                        
                        segment_parts.append(chunk_text)
                        
                        # 如果是區間最後一個且還在 focus 狀態，關閉標籤
                        if i == end_idx and is_in_focus:
                            segment_parts.append("[Focus Area End]")
                    
                    segments.append(" ".join(segment_parts))
                else:
                    # 沒有區間信息，直接使用 page_content
                    segments.append(doc.page_content)
            
            # 用分隔線組合該文件的不同區間
            full_content = "\n\n--- (中間內容省略) ---\n\n".join(segments)
            
            # 生成該文件的 XML document
            xml_doc = f"""<document id="{doc_idx}">
<source>{filename}</source>
<content>
{full_content}
</content>
</document>"""
            xml_documents.append(xml_doc)
        
        # 組合所有 document
        return "\n\n".join(xml_documents)
    
    def _build_chunk_content_cache(
        self, 
        chroma: Chroma, 
        results_by_source: dict
    ) -> dict:
        """
        從 Chroma 獲取所有需要的 chunk 內容並建立快取
        
        Args:
            chroma: Chroma 向量數據庫實例
            results_by_source: 按來源文件分組的檢索結果
        
        Returns:
            快取字典: {chunk_id: content}
            按 chunk_id 訪問內容
        """
        chunk_cache = {}
        
        try:
            # 收集所有需要的 parent_chunk_ids
            needed_chunk_ids = set()
            
            for source, file_results in results_by_source.items():
                for r in file_results:
                    # 收集此 result 的所有 parent_chunk_ids
                    parent_chunk_ids = r.get('parent_chunk_ids', [])
                    for chunk_id in parent_chunk_ids:
                        needed_chunk_ids.add(chunk_id)
            
            if not needed_chunk_ids:
                logger.warning("No parent_chunk_ids to cache")
                return chunk_cache
            
            # 從 Chroma 獲取所有需要的 chunk
            all_data = chroma.get(include=["documents", "metadatas"])
            
            for i, metadata in enumerate(all_data.get("metadatas", [])):
                chunk_id = metadata.get("chunk_id", "")
                
                # 只存儲需要的 chunk_id
                if chunk_id and chunk_id in needed_chunk_ids:
                    content = all_data["documents"][i]
                    chunk_cache[chunk_id] = content
            
            logger.info(f"Built chunk cache: {len(chunk_cache)} chunks")
            
        except Exception as e:
            logger.warning(f"Failed to build chunk cache: {str(e)}")
        
        return chunk_cache
    
    def _remove_overlap_between_chunks(self, text1: str, text2: str, min_overlap: int = 20) -> str:
        """
        移除兩個文本之間的 overlap 部分
        
        Args:
            text1: 第一個文本
            text2: 第二個文本（可能與 text1 的結尾有重疊）
            min_overlap: 最小 overlap 長度（字符數）
            
        Returns:
            移除 overlap 後的 text2
        """
        if not text1 or not text2:
            return text2
        
        # 從較長的 overlap 開始嘗試（從 text1 的後半部分開始）
        max_search_len = min(len(text1) // 2, len(text2) // 2, 500)  # 限制搜索範圍
        
        best_overlap_len = 0
        
        # 從長到短尋找最長的 overlap
        for overlap_len in range(max_search_len, min_overlap - 1, -1):
            text1_suffix = text1[-overlap_len:]
            text2_prefix = text2[:overlap_len]
            
            if text1_suffix == text2_prefix:
                best_overlap_len = overlap_len
                break
        
        # 如果找到 overlap，移除 text2 開頭的重複部分
        if best_overlap_len > 0:
            logger.info(f"Found overlap: {best_overlap_len} characters")
            return text2[best_overlap_len:]
        
        return text2
    
    def _assemble_parent_chunk_by_ids(
        self,
        chunk_ids: List[str],
        current_chunk_id: str | List[str],
        chunk_cache: dict
    ) -> str:
        """
        根據 chunk_ids 動態組裝 parent chunk 內容（含 overlap 移除）
        
        實現「螢光筆」標記原則：
        1. 按順序遍歷 parent chunk 的所有 chunk_id
        2. 只對 current_chunk_id（真正命中的）添加 [Focus Area] 標記
        3. 其他 chunk 作為上下文背景
        
        Args:
            chunk_ids: parent chunk 包含的 chunk_id 列表 [prev_id, current_id, next_id]
            current_chunk_id: 真正被向量命中的 chunk_id（螢光筆標記位置）
                            可以是單個 chunk_id (str) 或多個 chunk_ids (List[str])
            chunk_cache: chunk 內容快取 {chunk_id: content}
        
        Returns:
            組裝後的 parent chunk 內容（已移除重疊部分，精準標記命中點）
        """
        # 統一處理為 set 以便快速查找
        hit_chunk_ids = set([current_chunk_id]) if isinstance(current_chunk_id, str) else set(current_chunk_id)
        chunks = []
        original_chunks = []  # 保存原始內容用於 overlap 檢測
        chunk_roles = []  # 記錄每個 chunk 的角色
        
        for chunk_id in chunk_ids:
            # 直接從 cache 中通過 chunk_id 獲取內容
            content = chunk_cache.get(chunk_id, "")
            
            if not content:
                logger.warning(f"Missing content for chunk {chunk_id}")
                continue
            
            original_chunks.append(content)
            
            # 螢光筆標記：只標記真正被向量命中的 chunk
            is_hit = (chunk_id in hit_chunk_ids)
            if is_hit:
                content = f"[Focus Area Start]\n{content}\n[Focus Area End]"
                chunk_roles.append(f"Hit-{chunk_id}")
            else:
                chunk_roles.append(f"Context-{chunk_id}")
            
            chunks.append(content)
        
        # 移除相鄰 chunks 之間的 overlap
        if len(chunks) > 1:
            assembled_chunks = [chunks[0]]
            overlap_removed_count = 0
            total_overlap_chars = 0
            
            for i in range(1, len(chunks)):
                # 使用原始內容來檢測 overlap（避免 Focus Area 標記干擾）
                prev_original = original_chunks[i-1]
                curr_original = original_chunks[i]
                
                # 檢測並移除 overlap
                curr_without_overlap = self._remove_overlap_between_chunks(prev_original, curr_original)
                
                if len(curr_without_overlap) < len(curr_original):
                    overlap_removed_count += 1
                    overlap_chars = len(curr_original) - len(curr_without_overlap)
                    total_overlap_chars += overlap_chars
                    
                    # 如果當前 chunk 有 Focus Area 標記，需要重新添加
                    if chunk_ids[i] in hit_chunk_ids:
                        curr_without_overlap = f"[Focus Area Start]\n{curr_without_overlap}\n[Focus Area End]"
                    
                    assembled_chunks.append(curr_without_overlap)
                else:
                    assembled_chunks.append(chunks[i])
            
            assembled_content = '\n\n'.join(assembled_chunks)
            
            # 詳細日誌：顯示螢光筆標記效果
            hit_count = len([r for r in chunk_roles if r.startswith('Hit-')])
            context_count = len([r for r in chunk_roles if r.startswith('Context-')])
            
            if overlap_removed_count > 0:
                logger.info(
                    f"Assembled parent chunk by IDs: "
                    f"{len(chunks)} chunks ({hit_count} hits, {context_count} context), "
                    f"removed {overlap_removed_count} overlaps ({total_overlap_chars} chars), "
                    f"final length: {len(assembled_content)} chars"
                )
            else:
                logger.info(
                    f"Assembled parent chunk by IDs: "
                    f"{len(chunks)} chunks ({hit_count} hits, {context_count} context), "
                    f"{len(assembled_content)} chars"
                )
        else:
            # 只有一個 chunk，直接使用（必定是命中點）
            assembled_content = chunks[0] if chunks else ""
            logger.info(
                f"Assembled parent chunk by IDs: {len(chunks)} chunk (1 hit), {len(assembled_content)} chars"
            )
        
        return assembled_content
    
    def _merge_intervals(self, intervals: List[dict]) -> List[dict]:
        """
        區間合併算法
        
        將重疊或相鄰的區間合併，並選擇分數最高的結果作為代表
        
        Args:
            intervals: List[{start, end, data}] - 區間列表，data 包含原始檢索結果
        
        Returns:
            合併後的區間列表，每個區間包含 best_data（最佳代表）和 merged_count（合併數量）
        """
        if not intervals:
            return []
        
        # 按起始位置排序
        intervals.sort(key=lambda x: x['start'])
        
        merged = []
        curr_start = intervals[0]['start']
        curr_end = intervals[0]['end']
        curr_best_data = intervals[0]['data']
        curr_best_score = curr_best_data['score']
        curr_merged_count = 1
        
        for interval in intervals[1:]:
            next_start = interval['start']
            next_end = interval['end']
            next_data = interval['data']
            next_score = next_data['score']
            
            # 檢查是否重疊或相鄰 (next_start <= curr_end + 1)
            if next_start <= curr_end + 1:
                # 合併區間
                curr_end = max(curr_end, next_end)
                curr_merged_count += 1
                
                # 選擇分數更高的結果作為代表
                if next_score > curr_best_score:
                    curr_best_data = next_data
                    curr_best_score = next_score
            else:
                # 保存當前區間，開始新區間
                merged.append({
                    'start': curr_start,
                    'end': curr_end,
                    'best_data': curr_best_data,
                    'merged_count': curr_merged_count
                })
                curr_start = next_start
                curr_end = next_end
                curr_best_data = next_data
                curr_best_score = next_score
                curr_merged_count = 1
        
        # 保存最後一個區間
        merged.append({
            'start': curr_start,
            'end': curr_end,
            'best_data': curr_best_data,
            'merged_count': curr_merged_count
        })
        
        return merged

    def get_rag_context_with_parent_chunks(self, chroma: Chroma, collection_name: str, question: str, 
                                           k: int = 5, search_algorithm: Optional[str] = None, language: str = "zh-TW") -> Dict:
        """
        使用原始 RAG 方法：檢索 top-k chunks 並擴展為 parent chunks
        
        實現 Small-to-Big 檢索策略：
        1. 檢索 top-k 個 child chunks
        2. 擴展為 parent chunks 並去重
        3. 合併 parent chunks 作為 context
        
        Args:
            chroma: Chroma collection
            collection_name: 集合名稱
            question: 用戶問題
            k: 檢索的 top-k 數量
            search_algorithm: 搜尋演算法 ('semantic' 或 'bm25')
            language: 語言設定
        
        Returns:
            dict: {
                'include_file_list': List[str],  # 來源文件列表
                'chat_messages': List[dict],  # 推理的聊天消息列表
                'merged_content': str,  # 合併的 parent chunks 內容
                'retrieved_chunks': List[str],  # 檢索到的原始 chunks
                'error': str  # 錯誤信息，成功時為空字符串
            }
        """
        try:
            # 執行搜尋（使用共用方法）
            results = self._execute_search(
                chroma, collection_name, question, k, search_algorithm, language,
                log_prefix="[Parent Chunk RAG] "
            )

            # 構建 results_by_source 以供後續使用
            from collections import defaultdict
            results_by_source = defaultdict(list)
            for idx, (doc, score) in enumerate(results):
                source = doc.metadata.get('source', 'unknown')
                chunk_index = doc.metadata.get('chunk_index', 0)
                chunk_id = doc.metadata.get('chunk_id', f'unknown_{idx}')
                total_chunks = doc.metadata.get('total_chunks', 0)
                parent_chunk_ids_json = doc.metadata.get('parent_chunk_ids', '')
                
                parent_chunk_ids = []
                if parent_chunk_ids_json:
                    try:
                        parent_chunk_ids = json.loads(parent_chunk_ids_json)
                    except json.JSONDecodeError:
                        pass
                
                results_by_source[source].append({
                    'doc': doc,
                    'score': score,
                    'chunk_index': chunk_index,
                    'chunk_id': chunk_id,
                    'parent_chunk_ids': parent_chunk_ids,
                    'total_chunks': total_chunks
                })
            
            # 構建 chunk 內容快取
            chunk_content_cache = self._build_chunk_content_cache(chroma, results_by_source)
            
            # 擴展為 parent chunks 並去重（傳入 chroma 以支援動態組裝）
            expanded_results = self._expand_and_deduplicate_parent_chunks(results, chroma)
            
            if not expanded_results:
                raise ValueError('no parent chunks found after expansion')

            # 構建 XML 格式的 merged_content
            merged_content = self._build_xml_merged_content(expanded_results, chunk_content_cache)
            
            logger.info(f"[Parent Chunk RAG] Generated XML with {len(expanded_results)} expanded chunks, total length: {len(merged_content)} chars")

            # 收集所有原始 chunks（用於調試）
            all_chunks = [doc.page_content for doc, score in results]
            
            # 收集所有來源文件
            include_file_list = []
            source_files_set = set()
            for doc, score in expanded_results:
                source_file = doc.metadata.get('source_file', '')
                if source_file and source_file not in source_files_set:
                    source_files_set.add(source_file)
                    include_file_list.append(source_file)
            
            # 創建聊天消息（使用共用方法）
            chat_messages = self._build_chat_messages(merged_content, question, language)
            
            logger.info(f"[Parent Chunk RAG] Generated {len(chat_messages)} chat messages")
            logger.info(f"[Parent Chunk RAG] Source files: {include_file_list}")

            return {
                'include_file_list': include_file_list,
                'chat_messages': chat_messages,
                'merged_content': merged_content,
                'retrieved_chunks': all_chunks,
                'error': ''
            }
                
        except Exception as e:
            logger.error(f"[Parent Chunk RAG] Error: {str(e)}")
            return self._build_error_response(f'internal error: {str(e)}')

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
                            language: str = "zh-TW", rag_method: str = "kv_cache_reuse") -> Dict:
        """
        準備 RAG 查詢所需的 messages 和上下文資訊（單一職責：只負責 RAG 檢索和 messages 構建）
        
        Args:
            collection_name: 集合名稱
            query: 用戶問題
            k: 檢索的 top-k 數量
            language: 語言設定
            rag_method: RAG 方法 ('kv_cache_reuse' 或 'parent_chunk')
                - 'kv_cache_reuse': 使用 merged file（預設，適合 KV Cache 重用）
                - 'parent_chunk': 使用 top-k parent chunks（原始 RAG 方法）
        
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
            # 驗證 rag_method 參數
            valid_methods = ['kv_cache_reuse', 'parent_chunk']
            if rag_method not in valid_methods:
                logger.warning(f"Invalid rag_method '{rag_method}', defaulting to 'kv_cache_reuse'")
                rag_method = 'kv_cache_reuse'
            
            logger.info(f"[prepare_rag_messages] Using RAG method: {rag_method}")
            
            # 獲取 collection
            chroma = self._get_collection(collection_name)
            
            # 根據 rag_method 選擇不同的 RAG 方法
            if rag_method == 'parent_chunk':
                # 使用 parent chunks 方法（原始 RAG）
                result = self.get_rag_context_with_parent_chunks(
                    chroma, collection_name, query, k, 
                    search_algorithm=self.search_algorithm, 
                    language=language
                )
            else:
                # 使用 KV Cache Reuse 方法（merged file）
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
            # parent_chunk 方法不返回 filename，只有 kv_cache_reuse 方法會返回
            filename = result.get("filename", None)
            filename_wo_ext = os.path.splitext(filename)[0] if filename else None
            merged_content = result.get("merged_content", "")
            
            # 直接使用 RAG 方法中已經構建好的 chat_messages
            # 避免重複構建（get_rag_context_with_parent_chunks 和 get_rag_context_with_file_content 已經調用了 _build_chat_messages）
            messages = result.get("chat_messages", [])
            
            # 構建調試資訊
            debug_info = {
                "km_service_used": True,
                "collection": collection_name,
                "filename": filename_wo_ext,  # parent_chunk 方法時為 None
                "original_query": query,
                "rag_content_length": len(merged_content),
                "include_file_list": result.get("include_file_list", []),
                "rag_method": rag_method
            }
            
            return {
                'success': True,
                'messages': messages,
                'message': 'RAG messages prepared successfully',
                'merged_file': filename,  # parent_chunk 方法時為 None
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
                               params: Optional[Dict] = None, language: str = "zh-TW",
                               rag_method: str = "kv_cache_reuse") -> Dict:
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
            rag_method: RAG 方法 ('kv_cache_reuse' 或 'parent_chunk')
        
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
            # 使用 prepare_rag_messages 函數（支持 rag_method 參數）
            rag_result = self.prepare_rag_messages(collection_name, query, k, language, rag_method)
            
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
    logger.info(f"Test mode: using 64-dim fake embedding model, search algorithm: {rag_query_service.search_algorithm.upper()}")

    collections = rag_query_service.get_available_collections()
    logger.info(f"Available collections: {collections}")
    
    # 簡單的 RAG 查詢測試
    if collections:
        test_collection = collections[0]
        logger.info(f"\nTesting RAG query - Collection: {test_collection}")
        
        # # 先檢查 collection 狀態
        # try:
        #     chroma = rag_query_service._get_collection(test_collection)
        #     count = chroma._collection.count()
        #     logger.info(f"Collection document count: {count}")
        # except Exception as e:
        #     logger.info(f"Failed to get collection status: {str(e)}")
        
        result = rag_query_service.get_rag_context_with_file_content(
            collection_name=test_collection,
            question="what is NVM ExpressTM",
            k=3
        )
        
        if result.get('error'):
            logger.info(f"Query failed: {result['error']}")
        else:
            logger.info(f"Query succeeded")
            logger.info(f"   Recommended file: {result.get('filename', 'N/A')}")
            logger.info(f"   Message count: {len(result.get('chat_messages', []))}")
            # logger.info(result)
        
        # 測試 generate_openai_payload 功能
        logger.info(f"\n=== Testing OpenAI Payload Generation ===")
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
                logger.info(f"OpenAI Payload generated successfully")
                logger.info(f"   Message: {openai_result['message']}")
                logger.info(f"   Payload length: {len(openai_result['payload_raw'])} chars")
                
                # 顯示 payload 內容（前 500 字符）
                payload_preview = openai_result['payload_raw'][:500]
                logger.info(f"   Payload preview: {payload_preview}...")
                
                # 嘗試解析 JSON 來驗證格式
                try:
                    import json
                    payload_obj = json.loads(openai_result['payload_raw'])
                    logger.info(f"   JSON format validation passed")
                    logger.info(f"   Model: {payload_obj.get('model', 'N/A')}")
                    logger.info(f"   Stream: {payload_obj.get('stream', 'N/A')}")
                    logger.info(f"   Message count: {len(payload_obj.get('messages', []))}")
                except json.JSONDecodeError as e:
                    logger.info(f"   JSON format error: {str(e)}")
            else:
                logger.info(f"OpenAI Payload generation failed: {openai_result['message']}")
                
        except Exception as e:
            logger.info(f"OpenAI Payload test failed: {str(e)}")
    else:
        logger.info("\nNo available collections")

