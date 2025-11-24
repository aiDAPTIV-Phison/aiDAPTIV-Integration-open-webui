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
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys
import os
# 確保父目錄在 sys.path 中
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import settings
from kv_cache_content import KVcacheContentHandler

print(settings)
@dataclass
class ProcessingConfig:
    """文檔處理配置"""
    chunk_size: int = 512
    chunk_overlap: int = 20
    max_tokens_per_group: int = settings.MAX_TOKENS_PER_GROUP
    embedding_url: str = settings.EMBEDDING_URL
    llm_model_path: str = settings.LLM_MODEL_PATH
    llm_model_dir: str = settings.LLM_MODEL_DIR
    llm_model_gguf: str = settings.LLM_GGUF
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


class DocumentProcessor:
    """智能文檔處理器 - 簡化版主控制器"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None, base_folder: str = "./data"):
        self.config = config or ProcessingConfig()
        self.base_folder = base_folder
        self.collection_name = self.config.collection_name
        
        # 根據 collection_name 創建專屬文件夾
        self.collection_folder = os.path.join(self.base_folder, self.collection_name)
        self.merged_files_dir = os.path.join(self.collection_folder, "merged_files")
        
        # 確保所有目錄存在（目錄已在 task_manager 中創建，這裡只是確保）
        os.makedirs(self.merged_files_dir, exist_ok=True)
        
        # 更新配置中的路徑
        self.config.output_path = self.collection_folder
        
        # 初始化 KVcacheContentHandler 相關屬性
        self.kv_cache_handler = True
        self.tokenizer = AutoTokenizer.from_pretrained(gguf_file=self.config.llm_model_gguf, pretrained_model_name_or_path=self.config.llm_model_dir)
        logger.info(f"llm_tokenizer_path successfully initialized")
        
        logger.info(f"llm_model_path: {self.config.llm_model_path}")
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
            # 檢查 token 數量
            # total_tokens = self._check_total_tokens_all_file(documents)
            # logger.info(f"Token 數量檢查通過，總共: {total_tokens}")
            # 1. 保存原始檔案內容到 file_content 目錄（每個文檔單獨保存）
            # self._merge_and_save_file_contents(documents)
            
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
            
            # self._save_processing_result(result, file_content_dict)
            
            logger.info(f"文檔處理完成: {task_id}, 耗時: {processing_time:.2f}秒")
            #return result
            
        except Exception as e:
            logger.error(f"文檔處理失敗: {task_id}, 錯誤: {str(e)}")
            raise
    
    
    def _cleanup_merged_files_for_chunking(self):
        """
        清理 merged_files 目錄，清除完整文檔文件，保留 part 文件
        """
        try:
            if not os.path.exists(self.merged_files_dir):
                return
                
            # 統計並刪除完整文件
            part_files = []
            complete_files = []
            for filename in os.listdir(self.merged_files_dir):
                if filename.endswith('.txt'):
                    base_name = filename[:-4]  # 移除 .txt
                    if '_part' in base_name:
                        part_files.append(filename)
                    else:
                        complete_files.append(filename)
            
            # 刪除完整文件
            deleted_count = 0
            for filename in complete_files:
                file_path = os.path.join(self.merged_files_dir, filename)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.debug(f"已刪除完整文件: {filename}")
                except Exception as e:
                    logger.warning(f"刪除文件 {filename} 失敗: {str(e)}")
            
            logger.info(f"清理完成：刪除了 {deleted_count} 個完整文檔文件，保留了 {len(part_files)} 個 part 文件")
                
        except Exception as e:
            logger.error(f"清理 merged_files 目錄失敗: {str(e)}")
            # 不拋出異常，因為這不是關鍵操作
    
    def get_file_content_dir(self) -> str:
        """獲取 file_content 目錄路徑"""
        return self.file_content_dir
    
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
#        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(   
#            model_name="gpt-4o",
        # tokenizer = AutoTokenizer.from_pretrained(self.config.llm_tokenizer_path)
        # text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenize(
        #     tokenizer,
        #     chunk_size=self.config.chunk_size,
        #     chunk_overlap=self.config.chunk_overlap,
        #     separators=[
        #         "。",    # 中文句號
        #         ".",     # 英文句號
        #         "\n\n",
        #         "\n",
        #         "？",    # 中文問號
        #         "?",     # 英文問號
        #         "！",    # 中文驚嘆號
        #         "!",     # 英文驚嘆號
        #         "、",    # 中文破折 (常見於日文、中文)
        #         "，",    # 中文逗號
        #         "：",    # 中文冒號
        #         "；",    # 中文分號
        #         "…",    # 省略號
        #         "—",    # 破折號
        #         "ー",    # 日文長音/破折
        #         ":",     # 英文冒號
        #         ";",     # 英文分號
        #         ""       # 空字串，確保文字被完整分割
        #     ],
        # )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=[
                "。",    # Chinese full stop
                ".",     # period
                "\n\n",
                "\n",
                "？",    # Chinese question mark
                "?",     # question mark
                "！",    # Chinese exclamation mark
                "!",     # exclamation mark
                "、",    # ideographic comma (CJK)
                "，",    # Chinese comma
                "：",    # Chinese colon
                "；",    # Chinese semicolon
                "…",    # ellipsis
                "—",    # em dash
                "ー",    # Japanese long vowel mark
                ":",     # colon
                ";",     # semicolon
                ""       # empty string to ensure full split
            ],
        )

        chunked_documents = []
        chunk_counter_by_file = defaultdict(int)
        
        # 清理 merged_files 目錄中的完整文檔文件，只保留 part 文件
        # self._cleanup_merged_files_for_chunking()
        
        for doc in documents:
            filename = doc.metadata.get('source', 'unknown')
            chunks = text_splitter.split_documents([doc])
            print('self.config.chunk_size: ', self.config.chunk_size)

            for chunk in chunks:
                chunk_counter_by_file[filename] += 1
                chunk_id = f"{filename}_chunk_{chunk_counter_by_file[filename]}"
                
                # 保留原始文檔的所有 metadata，並添加新的 chunk 相關 metadata
                original_metadata = doc.metadata.copy()
                chunk.metadata.update(original_metadata)
                chunk.metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_counter_by_file[filename],
                    # 'source_file': filename,
                    # 'parent_content': doc.page_content  # 添加原始文檔內容作為 parent_content
                })
                chunked_documents.append(chunk)
                
                # 注意：不在這裡保存 part 文件，避免雙重命名問題
                # 分塊後的文件將在 KV Cache 處理過程中正確保存
        
        logger.info(f"分塊完成: {len(documents)} 個文檔 -> {len(chunked_documents)} 個分塊")
        
        # 將所有 chunks 保存為 JSON 格式
        # self._save_chunks_as_json(chunked_documents)
        
        return chunked_documents
    
    def _save_chunks_as_json(self, chunked_documents: List[Document]):
        """
        將所有 chunks 保存為 JSON 格式到 output 目錄
        
        Args:
            chunked_documents: 分塊後的文檔列表
        """
        try:
            # 轉換為 List[dict] 格式
            chunks_data = []
            for chunk in chunked_documents:
                chunk_dict = {
                    'page_content': chunk.page_content,
                    'metadata': chunk.metadata
                }
                chunks_data.append(chunk_dict)
            
            # 保存到 output 目錄下的 chunks.json
            chunks_file_path = os.path.join(self.output_dir, "chunks.json")
            
            with open(chunks_file_path, 'w') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"已保存 {len(chunks_data)} 個 chunks 到: {chunks_file_path}")
            
        except Exception as e:
            logger.error(f"保存 chunks.json 失敗: {str(e)}")
            raise
    
    def load_chunks_from_json(self) -> List[Document]:
        """
        從 output 目錄讀取 chunks.json 並轉換回 Document 列表
        
        Returns:
            Document 列表
        """
        chunks_file_path = os.path.join(self.output_dir, "chunks.json")
        
        try:
            if not os.path.exists(chunks_file_path):
                logger.warning(f"chunks.json 文件不存在: {chunks_file_path}")
                return []
            
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # 轉換回 Document 列表
            documents = []
            for chunk_dict in chunks_data:
                document = Document(
                    page_content=chunk_dict['page_content'],
                    metadata=chunk_dict['metadata']
                )
                documents.append(document)
            
            logger.info(f"成功從 {chunks_file_path} 讀取 {len(documents)} 個 chunks")
            return documents
            
        except Exception as e:
            logger.error(f"讀取 chunks.json 失敗: {str(e)}")
            return []
    
    def _create_vector_database(self, documents: List[Document], collection_name: str) -> Chroma:
        from langchain_openai import OpenAIEmbeddings
        import httpx

        """創建向量數據庫"""
        logger.info(f"開始創建向量數據庫，共 {len(documents)} 個文檔")
        
        # 初始化嵌入模型 - 使用 API 方式
        embedding_url = os.getenv("EMBEDDING_URL", self.config.embedding_url)
        # embeddings = HuggingFaceInferenceAPIEmbeddings(api_url=embedding_url, api_key="empty")
        
        # # http_client = httpx.Client(verify=False)
        embeddings = OpenAIEmbeddings(base_url=embedding_url, api_key="EMPTY", 
                                   tiktoken_enabled=False, check_embedding_ctx_length=False )

        # 使用內存向量數據庫，無需刪除現有數據庫
        
        # 批量處理文檔以避免內存問題
        batch_size = 32
        batch_num = (len(documents) + batch_size - 1) // batch_size
        vectorstore = None

        logger.info(f"開始批量處理，共 {batch_num} 批，每批 {batch_size} 個文檔")
        persist_directory = os.path.join(self.base_folder, "chroma_db")
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
    
    def _save_processing_result(self, result: ProcessingResult, file_content_dict: Dict[str, List[Dict[str, any]]]):
        """保存處理結果"""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存分組結果
        groups_file = output_path / f"{result.task_id}_groups.json"
        groups_data = [asdict(group) for group in result.groups]
        with open(groups_file, 'w', encoding='utf-8') as f:
            json.dump(groups_data, f, ensure_ascii=False, indent=2)
        
        # 保存合併後的文件
        merged_folder = output_path / f"merged_files"
        merged_folder.mkdir(exist_ok=True)
        
        for group in result.groups:
            merged_file = merged_folder / f"{group.representative_file}.txt"
            with open(merged_file, 'w', encoding='utf-8') as f:
                f.write(group.merged_content)
        
        # 保存 chunk.json（由 file_content_dict 轉換而來）
        try:
            chunks_data = []
            for source_file, chunks in (file_content_dict or {}).items():
                for chunk in chunks:
                    chunk_data = {
                        "page_content": chunk.get("content", ""),
                        "metadata": {
                            "source_file": source_file,
                            "source": chunk.get("metadata", {}).get("source", source_file),
                            "page": chunk.get("metadata", {}).get("page", "1_1"),
                            "parent_content": chunk.get("content", ""),
                            "page_key": chunk.get("metadata", {}).get("page_key", f"{source_file}-{chunk.get('chunk_index', 0)}"),
                            "start_page": chunk.get("metadata", {}).get("start_page", 1),
                            "end_page": chunk.get("metadata", {}).get("end_page", 1),
                            "chunk_id": chunk.get("chunk_id", ""),
                            "chunk_index": chunk.get("chunk_index", 0),
                            "token_count": chunk.get("token_count", 0),
                            "group_id": chunk.get("metadata", {}).get("group_id", ""),
                        }
                    }
                    chunks_data.append(chunk_data)

            chunk_json_file = output_path / "chunks.json"
            with open(chunk_json_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=4)
            logger.info(f"已保存 {len(chunks_data)} 個 chunks 到: {chunk_json_file}")
        except Exception as e:
            logger.error(f"保存 chunks.json 失敗: {str(e)}")
        
        # 保存摘要信息
        summary = {
            'task_id': result.task_id,
            'total_files': result.total_files,
            'total_chunks': result.total_chunks,
            'total_groups': result.total_groups,
            'processing_time': result.processing_time,
            'created_at': result.created_at.isoformat(),
            'config': asdict(self.config)
        }
        
        summary_file = output_path / f"{result.task_id}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"處理結果已保存到: {output_path}")
    
    def initialize_kv_cache_handler(self, vectorstore: Chroma, file_max_tokens: int = 10000) -> None:
        """
        初始化 KVcacheContentHandler
        
        Args:
            vectorstore: 已創建的 Chroma 向量數據庫
            file_max_tokens: 每個文件的最大 token 數，超過將被分割
        """
        try:
            # 初始化 tokenizer（如果還沒有的話）
            # if self.tokenizer is None:
            #     self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_tokenizer_path)
            #     logger.info("tokenizer load success")
            
            # 初始化 KVcacheContentHandler
            self.kv_cache_handler = KVcacheContentHandler(
                chroma=vectorstore,
                tokenizer=self.tokenizer,
                file_max_tokens=file_max_tokens
            )
            
            logger.info("KVcacheContentHandler 初始化完成")
            
        except Exception as e:
            logger.error(f"初始化 KVcacheContentHandler 失敗: {str(e)}")
            raise
    
    def process_with_kv_cache(self, save_folder_path: str = None, save_similarity_matrix: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        使用 KVcacheContentHandler 處理文檔並獲取相似度分組
        
        Args:
            save_folder_path: 保存路徑（可選），如果為 None 則不保存
            save_similarity_matrix: 是否保存相似度矩陣（預設 True）
            
        Returns:
            按相似度分組並擴展後的內容字典
        """
        if self.kv_cache_handler is None:
            raise ValueError("KVcacheContentHandler 未初始化，請先調用 initialize_kv_cache_handler")
        
        try:
            # 使用 KVcacheContentHandler 進行相似度分組處理
            grouped_content, similarity_matrix, filenames, original_content, file_content_dict = self.kv_cache_handler.group_and_save_similar_documents(save_folder_path)
            
            # 如果請求保存相似度矩陣並且有有效的矩陣資料
            if save_similarity_matrix and save_folder_path is not None and similarity_matrix.size > 0:
                # 從 original_content 中提取每個文件的 token 數量
                file_token_counts = {}
                for filename in filenames:
                    if filename in original_content:
                        file_token_counts[filename] = original_content[filename].get("total_token_count", 0)
                    else:
                        file_token_counts[filename] = 0
                
                self.kv_cache_handler.save_similarity_matrix(similarity_matrix, filenames, save_folder_path, file_token_counts)
            
            logger.info(f"KV Cache 處理完成，共產生 {len(grouped_content)} 個分組")
            return grouped_content, file_content_dict
            
        except Exception as e:
            logger.error(f"KV Cache 處理失敗: {str(e)}")
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
            return self.kv_cache_handler.prepare_kv_cache_content(save_folder_path)
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