"""
智能文檔處理服務
負責完整的文檔處理流程：
1. 接收外部解析器結果
2. 創建文檔分塊和向量數據庫
3. 計算文件級別的嵌入向量
4. 基於相似度合併文件
5. 存儲處理結果
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

from loguru import logger
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import os
from langchain_openai import OpenAIEmbeddings
import requests
# 確保父目錄在 sys.path 中
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import settings
from services.kv_cache_content import KVcacheContentHandler
from services.lib.tokenizer_manager import TokenCounter

# token_counter = TokenCounter(tokenizer_path=r"D:\AI\Code\agentbuilder\qwen3-embed\tokenizer.json")
# logger.info(settings)  # Commented out to avoid Windows cp1252 encoding error
@dataclass
class ProcessingConfig:
    """文檔處理配置"""
    chunk_size: int = 512
    chunk_overlap: int = 102
    max_tokens_per_group: int = settings.MAX_TOKENS_PER_GROUP
    embedding_model_name: str = settings.EMBEDDING_MODEL_NAME
    embedding_url: str = settings.EMBEDDING_API_URL
    embedding_type: str = settings.EMBEDDING_TYPE
    collection_name: str = "documents"
    output_path: str = "./processed_output"


@dataclass
class SimilarityGroup:
    """相似度分組結果"""
    group_id: str
    representative_file: str
    files_in_group: List[str]
    total_tokens: int
    average_similarity: float
    merged_content: str


@dataclass
class ProcessingResult:
    """處理結果"""
    task_id: str
    total_files: int
    total_chunks: int
    total_groups: int
    groups: List[SimilarityGroup]
    processing_time: float
    created_at: datetime

def count_tokens_embedding(text: str) -> int:
    API_URL = f"http://{settings.EMBEDDING_API_IP}:{settings.EMBEDDING_API_PORT}/tokenize"
    MODEL = settings.EMBEDDING_MODEL_NAME
    CONTENT = text
    if settings.EMBEDDING_TYPE == "llamacpp":
        # 發送請求
        payload = {
            "model": MODEL,
            "content": CONTENT
        }
    elif settings.EMBEDDING_TYPE == "vllm":
        payload = {
            "model": MODEL,
            "prompt": CONTENT
        }
    # logger.info(f"Count tokens: API_URL {API_URL} {MODEL} {payload.keys()}")
    response = requests.post(API_URL, json=payload, timeout=30)
    result = response.json()
    token_length = len(result['tokens']) + 2  # multilingal bos & eos
    # token_length = token_counter.count_tokens(text)
    return token_length

class DocumentProcessor:
    """智能文檔處理器 - 簡化版主控制器"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None, base_folder: str = "./data"):
        self.config = config or ProcessingConfig()
        self.base_folder = base_folder
        self.collection_name = self.config.collection_name
        
        # 根據 collection_name 創建專屬文件夾
        self.collection_folder = os.path.join(self.base_folder, self.collection_name)
        # merged_files 存儲在新的目錄結構中
        self.merged_files_dir = os.path.join(self.collection_folder, "merged_files")
        
        # 確保所有目錄存在（目錄已在 task_manager 中創建，這裡只是確保）
        os.makedirs(self.merged_files_dir, exist_ok=True)
        
        # 更新配置中的路徑
        self.config.output_path = self.collection_folder
        
        # 初始化 KVcacheContentHandler 相關屬性
        self.kv_cache_handler = True

        logger.info("DocumentProcessor initialization completed")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Collection folder: {self.collection_folder}")
        logger.info(f"Merged files directory: {self.merged_files_dir}")

    
    def process_documents(
        self, 
        documents: List[Document],
        task_id: str,
        collection_name: str,
        save_similarity_matrix: bool = True
    ) -> ProcessingResult:
        """
        完整的文檔處理流程
        
        Args:
            documents: 外部解析器提供的文檔列表
            task_id: 任務ID
            use_kv_cache: 是否使用 KV Cache 進行處理（預設 False）
            kv_cache_save_path: KV Cache 結果的保存路徑（可選）
            save_similarity_matrix: 是否保存相似度矩陣（預設 True）
            
        Returns:
            處理結果
        """
        start_time = datetime.now()
        logger.info(f"Starting document processing task: {task_id}")
        
        try:
            # # 2. 文檔分塊
            chunked_documents = self._chunk_documents(documents)
            
            # # 3. 創建向量數據庫
            vectorstore = self._create_vector_database(chunked_documents, collection_name)
            
            logger.info("Processing with KV Cache...")
            # 初始化 KVcacheContentHandler
            self.initialize_kv_cache_handler(vectorstore, self.config.max_tokens_per_group)
            
            # 使用 KV Cache 進行處理
            kv_cache_groups = self.get_kv_cache_content(self.merged_files_dir)
            
            # 將 KV Cache 結果轉換為標準分組格式
            groups_with_content = self._convert_kv_cache_to_groups(kv_cache_groups)
                
            # # 7. 保存處理結果
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ProcessingResult(
                task_id=task_id,
                total_files=len(documents),
                total_chunks=len(chunked_documents),
                total_groups=len(groups_with_content),
                groups=groups_with_content,
                processing_time=processing_time,
                created_at=start_time
            )
            
            logger.info(f"Document processing completed: {task_id}, took: {processing_time:.2f}s")
            return result, chunked_documents
            
        except Exception as e:
            logger.error(f"Document processing failed: {task_id}, error: {str(e)}")
            raise
    
    
    def list_merged_files(self) -> List[str]:
        """列出所有合併後的檔案（完整路徑）"""
        try:
            files = []
            # 使用 self.merged_files_dir 變數
            if os.path.exists(self.merged_files_dir):
                for file_name in os.listdir(self.merged_files_dir):
                    if file_name.endswith('.txt'):
                        files.append(os.path.join(self.merged_files_dir, file_name))
            else:
                logger.warning(f"merged_files directory does not exist: {self.merged_files_dir}")
                
            return sorted(files)
        except Exception as e:
            logger.error(f"Failed to list merged files: {str(e)}")
            return []

    def list_merged_filenames(self) -> List[str]:
        """列出所有合併後的檔案名（原始文件名，含原始擴展名）"""
        try:
            files = []
            # 使用 self.merged_files_dir 變數
            if os.path.exists(self.merged_files_dir):
                for file_name in os.listdir(self.merged_files_dir):
                    if file_name.endswith('.txt'):
                        # 移除最後的 .txt 擴展名，得到原始文件名
                        original_filename = file_name[:-4]  # 去掉 ".txt"
                        files.append(original_filename)
            else:
                logger.warning(f"merged_files directory does not exist: {self.merged_files_dir}")
                
            return sorted(files)
        except Exception as e:
            logger.error(f"Failed to list merged filenames: {str(e)}")
            return []
    

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """對文檔列表進行分塊"""
        logger.info(f"Local: starting to chunk {len(documents)} documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators = [
                # --- Level 1: 段落與結構 (最優先) ---
                r"\n\n",       # 雙換行
                r"\n",         # 單換行
                
                # --- Level 2: 句子結束 (語意完整) ---
                "。",          # 中文句號 (安全)
                r"\.(?!\d)",   # 智慧點號：避開小數點 (3.14)，但會切開句子
                "！",          # 中文驚嘆號
                r"\!",         # 英文驚嘆號 (建議跳脫以防萬一)
                "？",          # 中文問號
                r"\?",         # 英文問號 (絕對必須跳脫，否則 Crash)
                
                # --- Level 3: 子句與語氣 ---
                "；",          # 中文分號
                r";",          # 英文分號
                "：",          # 中文冒號
                r":",          # 英文冒號
                
                # --- Level 4: 短語與列表 ---
                r"\|",         # 直線符號 (絕對必須跳脫，否則視為 OR)
                "，",          # 中文逗號
                r",(?!\d)",    # 智慧逗號：避開千分位 (1,000)
                "、",          # 頓號
                
                # --- Level 5: 單字邊界 ---
                r"\s+",        # 匹配所有空白 (Space, Tab)，比 " " 更強大
                ""             # 最後手段
            ],
            length_function = count_tokens_embedding,
            is_separator_regex=True
        )

        chunked_documents = []
        chunk_counter_by_file = defaultdict(int)
        
        for doc in documents:
            filename = doc.metadata.get('source', 'unknown')
            chunks = text_splitter.split_documents([doc])
            logger.info('self.config.chunk_size: ', self.config.chunk_size)

            for chunk in chunks:
                chunk_counter_by_file[filename] += 1
                # INSERT_YOUR_CODE
                # 將 filename 中所有的點替換為 "_"
                if "." in filename:
                    filename = filename.replace(".", "_")
                chunk_id = f"{filename}_chunk_{chunk_counter_by_file[filename]}"
                
                # 保留原始文檔的所有 metadata，並添加新的 chunk 相關 metadata
                original_metadata = doc.metadata.copy()
                chunk.metadata.update(original_metadata)
                chunk.metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_counter_by_file[filename],
                })
                chunked_documents.append(chunk)
                
                # 注意：不在這裡保存 part 文件，避免雙重命名問題
                # 分塊後的文件將在 KV Cache 處理過程中正確保存
        
        logger.info(f"Chunking completed: {len(documents)} documents -> {len(chunked_documents)} chunks")
        
        return chunked_documents

    
    def _build_parent_chunk_map(self, documents: List[Document]) -> Dict[str, Dict]:
        """
        構建 parent chunk 映射表，為每個 chunk 儲存其上下文 chunk_ids
        
        特性：
        1. 只儲存 chunk_ids，不合併內容
        2. 查詢時再動態組裝內容
        3. 正確處理邊界情況（首尾 chunk、單一 chunk）
        
        Args:
            documents: 已分塊的文檔列表
            
        Returns:
            Dict[chunk_id, Dict]: chunk_id 到 parent chunk 信息的映射
            - chunk_ids: 組成 parent chunk 的 chunk_id 列表 [prev_id, current_id, next_id]
            - is_boundary: 是否為邊界 chunk（首尾）
            - parent_start_index: parent chunk 的起始索引
            - parent_end_index: parent chunk 的結束索引
            - total_chunks: 該文件的總 chunk 數
        """
        logger.info(f"Building parent chunk map for {len(documents)} chunks")
        
        # 按 source (filename) 分組
        docs_by_file = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            docs_by_file[source].append(doc)
        
        logger.info(f"File grouping completed, {len(docs_by_file)} files")
        
        # 為每個文件構建 parent chunk
        parent_map = {}
        total_boundary_chunks = 0
        total_single_chunks = 0
        
        for source, file_docs in docs_by_file.items():
            # 按 chunk_index 排序
            sorted_docs = sorted(file_docs, key=lambda d: d.metadata.get('chunk_index', 0))
            total_chunks_in_file = len(sorted_docs)
            
            # 獲取該文件的最小 chunk_index（用於邊界檢查）
            min_chunk_index = min(d.metadata.get('chunk_index', i + 1) for i, d in enumerate(sorted_docs))
            
            # logger.debug(f"處理文件 {source}，共 {total_chunks_in_file} 個 chunk，最小 chunk_index={min_chunk_index}")
            logger.debug(f"Processing file {source}, total {total_chunks_in_file} chunks, min chunk_index={min_chunk_index}")
            
            for i, doc in enumerate(sorted_docs):
                chunk_id = doc.metadata.get('chunk_id', f'unknown_{i}')
                chunk_index = doc.metadata.get('chunk_index', i + 1)  # chunk_index 從 1 開始（默認）
                chunk_ids = []
                
                # 計算 parent chunk 的索引範圍 (window_size = 1)
                parent_start_index = max(min_chunk_index, chunk_index - 1)
                parent_end_index = min(total_chunks_in_file, chunk_index + 1)
                
                # 前一個 chunk
                if i > 0:
                    chunk_ids.append(sorted_docs[i-1].metadata.get('chunk_id', f'prev_{i}'))
                
                # 當前 chunk
                chunk_ids.append(chunk_id)
                
                # 後一個 chunk
                if i < len(sorted_docs) - 1:
                    chunk_ids.append(sorted_docs[i+1].metadata.get('chunk_id', f'next_{i}'))
                
                # 判斷邊界
                is_boundary = (i == 0 or i == len(sorted_docs) - 1)
                if is_boundary:
                    total_boundary_chunks += 1
                
                # 判斷單一 chunk 文件
                if len(sorted_docs) == 1:
                    total_single_chunks += 1
                
                parent_map[chunk_id] = {
                    'chunk_ids': chunk_ids,
                    'is_boundary': is_boundary,
                    'parent_start_index': parent_start_index,
                    'parent_end_index': parent_end_index,
                    'total_chunks': total_chunks_in_file
                }
        
        logger.info(f"Parent chunk map built, {len(parent_map)} entries")
        logger.info(f"Stats: boundary chunks: {total_boundary_chunks}, single-chunk files: {total_single_chunks}")
        
        return parent_map
    
    def _create_vector_database(self, documents: List[Document], collection_name: str) -> Chroma:
        
        import httpx

        """創建向量數據庫"""
        logger.info(f"Creating vector database for {len(documents)} documents")
        
        # 初始化嵌入模型 - 使用 API 方式
        embedding_url = os.getenv("EMBEDDING_URL", self.config.embedding_url)
        logger.info(f"embedding_url: {embedding_url}")
        
        if self.config.embedding_type == "tei":
            embeddings = HuggingFaceInferenceAPIEmbeddings(api_url=embedding_url, api_key="empty")
            logger.info(f"Using TEI embedding model")
        elif self.config.embedding_type == "vllm" or self.config.embedding_type == "openai" or self.config.embedding_type == "llamacpp":
            embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model_name,
                base_url=embedding_url,
                api_key="EMPTY",
                tiktoken_enabled=False,
                check_embedding_ctx_length=False,
                encoding_format="float"
            )
            logger.info(f"Using OpenAI format embedding model")

        # 使用內存向量數據庫，無需刪除現有數據庫
        
        # 構建 parent chunk 映射表（Small-to-Big 策略）
        logger.info("Building parent chunk map...")
        parent_chunk_map = self._build_parent_chunk_map(documents)
        
        # 批量處理文檔以避免內存問題
        batch_size = 1
        batch_num = (len(documents) + batch_size - 1) // batch_size
        vectorstore = None

        logger.info(f"Starting batch processing, {batch_num} batches, {batch_size} documents per batch")
        # 每個 collection 有獨立的 chromadb 目錄
        persist_directory = os.path.join(self.base_folder, collection_name, "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)
        for batch_idx in range(batch_num):
            start_index = batch_idx * batch_size
            end_index = min((batch_idx + 1) * batch_size, len(documents))
            batch_documents = documents[start_index:end_index]
            
            logger.info(f"Processing batch {batch_idx + 1}/{batch_num}, contains {len(batch_documents)} documents")

            # 為每個 batch 生成 IDs 並添加 parent chunk metadata
            batch_ids = []
            for i, doc in enumerate(batch_documents):
                # 優先使用 chunk_id 作為 ID
                chunk_id = doc.metadata.get('chunk_id', None)
                if chunk_id:
                    batch_ids.append(chunk_id)
                else:
                    # 如果沒有 chunk_id，使用批次索引和文檔索引生成唯一 ID
                    global_index = start_index + i
                    batch_ids.append(f"doc_{global_index}")
                    chunk_id = f"doc_{global_index}"
                
                # 添加 parent chunk 信息到 metadata
                parent_info = parent_chunk_map.get(chunk_id, {})
                if parent_info:
                    chunk_ids_list = parent_info.get('chunk_ids', [])
                    
                    # 將 parent chunk 信息添加到 metadata
                    # 注意：Chroma 不支持 list 類型的 metadata，需要轉換為 JSON 字符串
                    doc.metadata['parent_chunk_ids'] = json.dumps(chunk_ids_list) if chunk_ids_list else ""
                    doc.metadata['is_boundary'] = parent_info.get('is_boundary', False)
                    
                    # 添加 parent 區間索引信息（用於查詢時動態組裝）
                    doc.metadata['parent_start_index'] = parent_info.get('parent_start_index', 0)
                    doc.metadata['parent_end_index'] = parent_info.get('parent_end_index', 0)
                    doc.metadata['total_chunks'] = parent_info.get('total_chunks', 0)
                    
                    logger.debug(
                        f"Added parent chunk metadata to {chunk_id}: "
                        f"chunk_ids={chunk_ids_list}, "
                        f"range=[{parent_info.get('parent_start_index')}, {parent_info.get('parent_end_index')}]"
                    )
                else:
                    logger.warning(f"No parent chunk info found for {chunk_id}")
            
            if batch_idx == 0:
                logger.info(f"embedding: {embeddings}, embedding_url: {embedding_url}")
                # 第一批：創建新的向量存儲（僅內存）
                vectorstore = Chroma.from_documents(
                    documents=batch_documents,
                    embedding=embeddings,
                    collection_name=collection_name,
                    ids=batch_ids,
                    persist_directory=persist_directory
                )
            else:
                # 後續批次：添加到現有向量存儲，指定 ids
                vectorstore.add_documents(
                    documents=batch_documents,
                    ids=batch_ids
                )
            
            logger.info(f"Batch {batch_idx + 1} completed, added {len(batch_documents)} documents (IDs: {batch_ids[:3]}...)")
        vectorstore.persist()
        logger.info(f"In-memory vector database created, processed {len(documents)} documents")
        return vectorstore
    
    def initialize_kv_cache_handler(self, vectorstore: Chroma, file_max_tokens: int = 10000) -> None:
        """
        初始化 KVcacheContentHandler
        
        Args:
            vectorstore: 已創建的 Chroma 向量數據庫
            file_max_tokens: 每個文件的最大 token 數，超過將被分割
        """
        try:
            # 初始化 KVcacheContentHandler
            self.kv_cache_handler = KVcacheContentHandler(
                chroma=vectorstore,
                file_max_tokens=file_max_tokens
            )
            
            logger.info("KVcacheContentHandler initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize KVcacheContentHandler: {str(e)}")
            raise
    
    def get_kv_cache_content(self, save_folder_path: str = None) -> Dict[str, Dict[str, Any]]:
        """
        獲取 KV Cache 格式的內容
        
        Returns:
            KV Cache 格式的內容字典
        """
        if self.kv_cache_handler is None:
            raise ValueError("KVcacheContentHandler 未初始化，請先調用 initialize_kv_cache_handler")
        
        try:
            return self.kv_cache_handler.process_all_documents(save_folder_path = save_folder_path)
        except Exception as e:
            logger.error(f"Failed to get KV Cache content: {str(e)}")
            raise
    
    def _convert_kv_cache_to_groups(self, kv_cache_groups: Dict[str, Dict[str, Any]]) -> List[SimilarityGroup]:
        """
        將 KV Cache 處理結果轉換為標準的 SimilarityGroup 格式
        
        Args:
            kv_cache_groups: KV Cache 處理後的分組結果
            
        Returns:
            轉換後的 SimilarityGroup 列表
        """
        groups = []
        
        for group_idx, (representative_file, group_data) in enumerate(kv_cache_groups.items()):
            # 從 group_data 中提取信息
            content = group_data.get("content", "")
            total_tokens = group_data.get("total_token_count", 0)
            group_files = group_data.get("group_files", [representative_file])
            
            # 使用真實的平均相似度，如果沒有則預設為 1.0
            average_similarity = group_data.get("average_similarity", 1.0)
            
            # 創建 SimilarityGroup
            group = SimilarityGroup(
                group_id=f"kv_cache_group_{group_idx}",
                representative_file=representative_file,
                files_in_group=group_files,
                total_tokens=total_tokens,
                average_similarity=average_similarity,
                merged_content=content
            )
            
            groups.append(group)
        
        logger.info(f"Converted {len(kv_cache_groups)} KV Cache groups to standard format")
        return groups 