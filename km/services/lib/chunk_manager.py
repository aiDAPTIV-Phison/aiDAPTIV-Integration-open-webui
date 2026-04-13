"""
Chunk 管理模組
負責文檔的 chunk 切分和 token 計算相關功能
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict

from loguru import logger
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

# 確保父目錄在 sys.path 中
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import settings


def count_tokens_embedding(text: str) -> int:
    """
    計算文本的 token 數量（使用 embedding API）
    
    Args:
        text: 要計算 token 數的文本
        
    Returns:
        token 數量
    """
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
    return len(result['tokens'])


class ChunkManager:
    """Chunk 管理器 - 負責文檔的切分和 token 計算"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        max_tokens_per_group: int = None
    ):
        """
        初始化 ChunkManager
        
        Args:
            chunk_size: 每個 chunk 的大小（預設 500）
            chunk_overlap: chunk 之間的重疊大小（預設 100）
            max_tokens_per_group: 每個 group 的最大 token 數（預設使用 settings.MAX_TOKENS_PER_GROUP）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens_per_group = max_tokens_per_group or settings.MAX_TOKENS_PER_GROUP
        
        logger.info(f"ChunkManager initialization completed")
        logger.info(f"chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}")
        logger.info(f"max_tokens_per_group: {self.max_tokens_per_group}")
    
    def count_tokens(self, text: str) -> int:
        """
        計算文本的 token 數量
        
        Args:
            text: 要計算 token 數的文本
            
        Returns:
            token 數量
        """
        return count_tokens_embedding(text)
    
    def get_file_token_counts(self, documents: List[Document]) -> Dict[str, int]:
        """
        獲取檔案的 token 數
        
        Args:
            documents: 文檔列表
            
        Returns:
            檔名 -> token 數的對應字典
        """
        file_token_counts = {}
        for doc in documents:
            filename = doc.metadata.get('source', 'unknown')
            total_tokens = count_tokens_embedding(doc.page_content)
            file_token_counts[filename] = total_tokens
        return file_token_counts
    
    def classify_documents_by_size(
        self, 
        documents: List[Document]
    ) -> Tuple[List[Document], List[Document], Dict[str, int]]:
        """
        根據檔案總 token 數分類為大檔案或小檔案
        
        Args:
            documents: 原始文件列表
            
        Returns:
            tuple: (large_files, small_files, file_token_counts)
                - large_files: 超過 max_tokens_per_group 的檔案列表
                - small_files: 未超過 max_tokens_per_group 的檔案列表
                - file_token_counts: 檔名 -> token 數的對應字典
        """
        logger.info(f"Starting file classification, {len(documents)} files")
        logger.info(f"max_tokens_per_group: {self.max_tokens_per_group}")
        
        large_files = []
        small_files = []
        file_token_counts = {}
        
        for doc in documents:
            filename = doc.metadata.get('source', 'unknown')
            
            # 計算檔案的總 token 數
            try:
                total_tokens = count_tokens_embedding(doc.page_content)
                file_token_counts[filename] = total_tokens
                
                logger.info(f"File {filename}: {total_tokens} tokens")
                
                # 判斷是否超過 max_tokens_per_group
                if total_tokens > self.max_tokens_per_group:
                    large_files.append(doc)
                    logger.info(f"File {filename} classified as large file ({total_tokens} > {self.max_tokens_per_group})")
                else:
                    small_files.append(doc)
                    logger.info(f"File {filename} classified as small file ({total_tokens} <= {self.max_tokens_per_group})")
                    
            except Exception as e:
                logger.error(f"Failed to calculate token count for file {filename}: {str(e)}")
                # 發生錯誤時，預設歸類為小檔案（較安全）
                small_files.append(doc)
                file_token_counts[filename] = 0
        
        logger.info(f"File classification completed: {len(large_files)} large files, {len(small_files)} small files")
        
        return large_files, small_files, file_token_counts
    
    def chunk_documents(
        self, 
        documents: List[Document], 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None, 
        length_function: Callable = count_tokens_embedding
    ) -> List[Document]:
        """
        對文檔列表進行分塊
        
        Args:
            documents: 要分塊的文檔列表
            chunk_size: chunk 大小（預設使用 self.chunk_size）
            chunk_overlap: chunk 重疊大小（預設使用 self.chunk_overlap）
            length_function: 計算長度的函數（預設使用 count_tokens_embedding）
            
        Returns:
            分塊後的文檔列表
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        logger.info(f"Starting to chunk {len(documents)} documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
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
            length_function=length_function,
            is_separator_regex=True
        )

        chunked_documents = []
        chunk_counter_by_file = defaultdict(int)

        for doc in documents:
            filename = doc.metadata.get('source', 'unknown')
            chunks = text_splitter.split_documents([doc])
            logger.info(f'chunk_size: {chunk_size}')

            for chunk in chunks:
                chunk_counter_by_file[filename] += 1
                # 將 filename 中最後一個點（即副檔名的點）替換為 "_"
                safe_filename = filename
                if "." in filename:
                    name_parts = filename.rsplit(".", 1)
                    safe_filename = f"{name_parts[0]}_{name_parts[1]}"
                chunk_id = f"{safe_filename}_chunk_{chunk_counter_by_file[filename]}"
                
                # 保留原始文檔的所有 metadata，並添加新的 chunk 相關 metadata
                original_metadata = doc.metadata.copy()
                chunk.metadata.update(original_metadata)
                chunk.metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_counter_by_file[filename],
                })
                chunked_documents.append(chunk)

        logger.info(f"Chunking completed: {len(documents)} documents -> {len(chunked_documents)} chunks")
        
        return chunked_documents
    
    def split_large_file_into_group_files(
        self,
        doc: Document,
        total_tokens: int,
        max_tokens_per_group: Optional[int] = None
    ) -> List[Document]:
        """
        將大檔案切分為多個 group file
        
        Args:
            doc: 原始大檔案 Document
            total_tokens: 檔案的總 token 數
            max_tokens_per_group: 每個 group file 的最大 token 數（預設使用 self.max_tokens_per_group）
            
        Returns:
            List[Document]: 切分後的 group file Document 列表
        """
        max_tokens_per_group = max_tokens_per_group or self.max_tokens_per_group
        
        filename = doc.metadata.get('source', 'unknown')
        logger.info(f"Starting to split large file {filename}, total tokens: {total_tokens}")
        
        # 處理檔名（將最後一個點替換為底線）
        safe_filename = filename
        if "." in filename:
            name_parts = filename.rsplit(".", 1)
            safe_filename = f"{name_parts[0]}_{name_parts[1]}"
        
        # 使用 RecursiveCharacterTextSplitter 切分為 group files
        # chunk_size 設為 max_tokens_per_group，chunk_overlap 設為 0（group file 之間不重疊）
        group_file_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens_per_group,
            chunk_overlap=0,  # group file 之間不重疊
            separators=[
                r"\n\n",       # 雙換行
                r"\n",         # 單換行
                "。",          # 中文句號
                r"\.(?!\d)",   # 智慧點號
                "！",          # 中文驚嘆號
                r"\!",         # 英文驚嘆號
                "？",          # 中文問號
                r"\?",         # 英文問號
                "；",          # 中文分號
                r";",          # 英文分號
                "：",          # 中文冒號
                r":",          # 英文冒號
                r"\|",         # 直線符號
                "，",          # 中文逗號
                r",(?!\d)",    # 智慧逗號
                "、",          # 頓號
                r"\s+",        # 空白
                ""             # 最後手段
            ],
            length_function=count_tokens_embedding,
            is_separator_regex=True
        )
        
        # 切分為 group files
        group_file_chunks = group_file_splitter.split_documents([doc])
        logger.info(f"Large file {filename} split into {len(group_file_chunks)} group files")
        
        # 為每個 group file 建立 Document 並添加 metadata
        group_file_documents = []
        for idx, group_chunk in enumerate(group_file_chunks):
            group_file_id = f"{safe_filename}_group_{idx + 1}"
            
            # 建立新的 Document，保留原始 metadata
            group_file_doc = Document(
                page_content=group_chunk.page_content,
                metadata=doc.metadata.copy()
            )
            
            # 添加 group file 相關 metadata
            group_file_doc.metadata.update({
                'group_file_id': group_file_id,
                'is_group_file': True,
                'original_source': filename,
                'group_file_index': idx + 1,
                'total_group_files': len(group_file_chunks)
            })
            
            group_file_documents.append(group_file_doc)
            logger.info(f"Created group file: {group_file_id}, token count: {count_tokens_embedding(group_chunk.page_content)}")
        
        return group_file_documents

