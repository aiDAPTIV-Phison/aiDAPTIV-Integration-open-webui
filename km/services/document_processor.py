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
import re
import requests
# 確保父目錄在 sys.path 中
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import settings
from services.kv_cache_content import KVcacheContentHandler

# logger.info(settings)  # 註釋掉以避免 Windows cp1252 編碼錯誤
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
    base_url = re.sub(r'/v\d+$', '', settings.EMBEDDING_URL)
    API_URL = f"{base_url}/tokenize"
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

        logger.info("DocumentProcessor 初始化完成")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Collection 文件夾: {self.collection_folder}")
        logger.info(f"合併檔案目錄: {self.merged_files_dir}")

    
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
        logger.info(f"開始處理文檔任務: {task_id}")
        
        try:
            # # 2. 文檔分塊
            chunked_documents = self._chunk_documents(documents)
            
            # # 3. 創建向量數據庫
            vectorstore = self._create_vector_database(chunked_documents, collection_name)
            
            logger.info("使用 KV Cache 進行處理...")
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
            
            logger.info(f"文檔處理完成: {task_id}, 耗時: {processing_time:.2f}秒")
            return result, chunked_documents
            
        except Exception as e:
            logger.error(f"文檔處理失敗: {task_id}, 錯誤: {str(e)}")
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
                logger.warning(f"merged_files 目錄不存在: {self.merged_files_dir}")
                
            return sorted(files)
        except Exception as e:
            logger.error(f"列出合併檔案失敗: {str(e)}")
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
                logger.warning(f"merged_files 目錄不存在: {self.merged_files_dir}")
                
            return sorted(files)
        except Exception as e:
            logger.error(f"列出合併檔案名失敗: {str(e)}")
            return []
    

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """對文檔列表進行分塊"""
        logger.info(f"Local 開始對 {len(documents)} 個文檔進行分塊")

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
        
        logger.info(f"分塊完成: {len(documents)} 個文檔 -> {len(chunked_documents)} 個分塊")
        
        return chunked_documents
    
    def _create_vector_database(self, documents: List[Document], collection_name: str) -> Chroma:
        
        import httpx

        """創建向量數據庫"""
        logger.info(f"開始創建向量數據庫，共 {len(documents)} 個文檔")
        
        # 初始化嵌入模型 - 使用 API 方式
        embedding_url = os.getenv("EMBEDDING_URL", self.config.embedding_url)
        logger.info(f"embedding_url: {embedding_url}")
        
        if self.config.embedding_type == "tei":
            embeddings = HuggingFaceInferenceAPIEmbeddings(api_url=embedding_url, api_key="empty")
            logger.info(f"使用 tei 嵌入模型")
        elif self.config.embedding_type == "vllm" or self.config.embedding_type == "openai" or self.config.embedding_type == "llamacpp":
            embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model_name,
                base_url=embedding_url,
                api_key="EMPTY",
                tiktoken_enabled=False,
                check_embedding_ctx_length=False,
                encoding_format="float"
            )
            logger.info(f"使用 openai format 嵌入模型")

        # 使用內存向量數據庫，無需刪除現有數據庫
        
        # 批量處理文檔以避免內存問題
        batch_size = 1
        batch_num = (len(documents) + batch_size - 1) // batch_size
        vectorstore = None

        logger.info(f"開始批量處理，共 {batch_num} 批，每批 {batch_size} 個文檔")
        # 每個 collection 有獨立的 chromadb 目錄
        persist_directory = os.path.join(self.base_folder, collection_name, "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)
        for batch_idx in range(batch_num):
            start_index = batch_idx * batch_size
            end_index = min((batch_idx + 1) * batch_size, len(documents))
            batch_documents = documents[start_index:end_index]
            
            logger.info(f"處理第 {batch_idx + 1}/{batch_num} 批，包含 {len(batch_documents)} 個文檔")

            # 為每個 batch 生成 IDs
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
            
            logger.info(f"第 {batch_idx + 1} 批處理完成，已添加 {len(batch_documents)} 個文檔 (IDs: {batch_ids[:3]}...)")
        vectorstore.persist()
        logger.info(f"內存向量數據庫創建完成，共處理 {len(documents)} 個文檔")
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
            
            logger.info("KVcacheContentHandler 初始化完成")
            
        except Exception as e:
            logger.error(f"初始化 KVcacheContentHandler 失敗: {str(e)}")
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
            logger.error(f"獲取 KV Cache 內容失敗: {str(e)}")
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
        
        logger.info(f"已將 {len(kv_cache_groups)} 個 KV Cache 分組轉換為標準格式")
        return groups 