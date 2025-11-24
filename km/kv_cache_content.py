"""
ChromaDB 內容檢索與處理模組

此模組提供 KVcacheContentHandler 類別，用於連接 ChromaDB 實例，
檢索文件內容，並按來源文件進行組織和適當的頁面排序。
"""

import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import chromadb
import numpy as np
import pandas as pd

from chromadb.config import Settings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document
from transformers import AutoTokenizer


def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """
    使用 tokenizer 計算文本中的 token 數量
    
    參數:
        text: 要計算 token 的文本
        tokenizer: 使用的 tokenizer
        
    返回:
        文本中的 token 數量
    """
    return len(tokenizer.encode(text))


class KVcacheContentHandler:
    """
    ChromaDB 內容檢索與處理類別
    
    此類別處理從 ChromaDB 實例檢索文件內容，
    並按來源文件進行組織和適當的頁面排序。
    """
    
    def __init__(self, 
                 chroma: Chroma, 
                 tokenizer: AutoTokenizer = None,
                 file_max_tokens: int = 16000):
        """
        初始化 KVcacheContentHandler
        
        參數:
            chroma: Chroma 向量存儲實例
            tokenizer: 用於計算 token 的 tokenizer（可選）
            file_max_tokens: 每個文件的最大 token 數，超過將被分割
        """
        self.chroma = chroma
        self.tokenizer = tokenizer
        self.file_max_tokens = file_max_tokens
    
    def _parse_filename(self, filename: str) -> Tuple[str, int]:
        """
        解析文件名，提取基礎文件名和 part 編號
        
        Args:
            filename: 文件名，格式如 "document_part1.txt" 或 "document_part1"
            
        Returns:
            (base_filename, part_number) 元組
        """
        # 使用正則表達式匹配 _part數字 的模式，.txt 後綴可選
        match = re.match(r'(.+)_part(\d+)(?:\.txt)?$', filename)
        if match:
            base_filename = match.group(1)
            part_number = int(match.group(2))
            return base_filename, part_number
        else:
            # 如果沒有匹配到 part 模式，則整個文件名作為基礎文件名，part 編號為 0
            base_filename = filename.replace('.txt', '') if filename.endswith('.txt') else filename
            return base_filename, 0
    
    def _ensure_directory_permissions(self, directory_path: str) -> None:
        """
        確保目錄擁有正確的權限 (777)，讓所有用戶都可以讀寫執行
        
        參數:
            directory_path: 目錄路徑
        """
        try:
            os.makedirs(directory_path, exist_ok=True, mode=0o777)
            os.chmod(directory_path, 0o777)
        except OSError as e:
            print(f"設置目錄權限失敗: {directory_path}, 錯誤: {e}")
    
    def _ensure_file_permissions(self, file_path: str) -> None:
        """
        確保文件擁有正確的權限 (777)，讓所有用戶都可以讀寫執行
        
        參數:
            file_path: 文件路徑
        """
        try:
            if os.path.exists(file_path):
                os.chmod(file_path, 0o777)
        except OSError as e:
            print(f"設置文件權限失敗: {file_path}, 錯誤: {e}")
    
    
    def get_chunk_content_from_chroma(self) -> Dict[str, List[Dict[str, any]]]:
        """
        從 ChromaDB 檢索 chunk 內容並按來源文件和 chunk_index 組織
        
        返回:
            映射來源文件名到包含 chunk 索引、內容和 token 數量的內容字典列表
        """
        data = self.chroma.get(include=["embeddings", "metadatas", "documents"])
        
        file_content_dict: Dict[str, List[Dict[str, any]]] = defaultdict(list)
        
        for idx in range(len(data["metadatas"])):
            metadata = data["metadatas"][idx]
            source_file = metadata["source"]
            chunk_content = data["documents"][idx] if "documents" in data and idx < len(data["documents"]) else ""
            
            # 獲取 chunk_index，如果沒有則設為 0
            chunk_index = metadata.get("chunk_index", 0)
            chunk_id = metadata.get("chunk_id", f"{source_file}_part{chunk_index}")
            
            # 如果有 tokenizer 則計算 token 數量
            token_count = 0
            token_count = count_tokens(chunk_content, self.tokenizer)


            # 建立內容字典
            content_dict = {
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
                "content": chunk_content,
                "token_count": token_count,
                "embeddings": data["embeddings"][idx],
                "chroma_id": data["ids"][idx],
                "metadata": metadata
            }
            
            file_content_dict[source_file].append(content_dict)
        
        # 對每個文件的 chunks 按 chunk_index 排序
        for source_file in file_content_dict:
            file_content_dict[source_file].sort(key=lambda x: x["chunk_index"])
        
        return file_content_dict
    
    def build_complete_files_with_chunk_splitting(
        self, file_content_dict: Dict[str, List[Dict[str, any]]],
        update_metadata: bool = False,
        extract_remaining: bool = True
    ) -> tuple[Dict[str, Dict[str, any]], List[Dict[str, any]]]:
        """
        透過按順序連接 chunk 內容來建立完整的文件內容
        如果文件超過 max_tokens，將被分割成多個部分
        最後剩餘的零碎 chunks 會被提取出來用於相似度合併
        
        參數:
            file_content_dict: 映射來源文件到 chunk 內容列表的字典
            update_metadata: 是否更新 ChromaDB 中的 metadata
            extract_remaining: 是否提取剩餘的零碎 chunks
        返回:
            (完整文件字典, 剩餘的零碎 chunks 列表) 元組
        """
        file_whole_content: Dict[str, Dict[str, any]] = {}
        all_remaining_chunks: List[Dict[str, any]] = []
        
        for source_file, content_list in file_content_dict.items():
            # 按 chunk_index 排序（應該已經排序過了，但再確保一次）
            content_list.sort(key=lambda x: x["chunk_index"])
            
            # 如果超過最大 token 則分割文件
            split_files, remaining_chunks = self._split_large_file_into_chunk_parts(
                source_file, content_list, update_metadata, extract_remaining
            )
            
            file_whole_content.update(split_files)
            
            # 收集剩餘的 chunks
            if extract_remaining and remaining_chunks:
                all_remaining_chunks.extend(remaining_chunks)
        
        return file_whole_content, all_remaining_chunks
    
    def _split_large_file_into_chunk_parts(
        self, source_file: str, content_list: List[Dict[str, any]], 
        update_metadata: bool = True,
        extract_remaining: bool = True
    ) -> tuple[Dict[str, Dict[str, any]], List[Dict[str, any]]]:
        """
        根據 token 限制將 chunk 列表分割成多個部分
        如果啟用 extract_remaining，最後剩餘的零碎 chunks 會被提取出來
        
        參數:
            source_file: 原始來源文件名
            content_list: chunk 內容字典列表
            update_metadata: 是否更新 ChromaDB 中的 metadata
            extract_remaining: 是否提取剩餘的零碎 chunks
            
        返回:
            tuple: (包含分割文件部分的字典, 剩餘的零碎 chunks 列表)
        """
        result = {}
        remaining_chunks = []
        current_part = 1
        current_chunks = []
        current_tokens = 0
        
        for idx, chunk_data in enumerate(content_list):
            chunk_tokens = chunk_data["token_count"]
            
            # 特殊情況：如果單個 chunk 就超過限制，記錄警告但仍然處理
            if chunk_tokens > self.file_max_tokens:
                from loguru import logger
                logger.warning(f"Chunk {chunk_data.get('chunk_id', 'unknown')} 的 token 數 ({chunk_tokens}) 超過限制 ({self.file_max_tokens})，但仍將其添加到分組中")
            
            # 如果添加此 chunk 將超過限制，則保存當前部分並開始新部分
            if current_tokens + chunk_tokens > self.file_max_tokens and current_chunks:
                # 保存當前部分
                file_key = f"{source_file}_merged_part{current_part}" if current_part > 1 or len(content_list) > len(current_chunks) else source_file
                
                complete_content = "\n\n".join([chunk["content"] for chunk in current_chunks])
                
                # 收集 ChromaDB ID
                chroma_ids = [chunk["chroma_id"] for chunk in current_chunks]
                
                # 更新 ChromaDB metadata 中的 group_id
                if update_metadata:
                    self._update_document_group_metadata(chroma_ids, file_key)
                
                result[file_key] = {
                    "content": complete_content,
                    "total_token_count": current_tokens,
                    "chroma_ids": chroma_ids,
                    "chunk_ids": [chunk["chunk_id"] for chunk in current_chunks]
                }
                
                # 開始新部分
                current_part += 1
                current_chunks = []
                current_tokens = 0
            
            # 將當前 chunk 添加到當前部分
            current_chunks.append(chunk_data)
            current_tokens += chunk_tokens
        
        # 處理最後一部分
        if current_chunks:
            # 檢查這部分是否應該被視為零碎剩餘
            # 如果這部分的 token 數少於一定閾值（例如 50%），視為零碎剩餘
            utilization_rate = current_tokens / self.file_max_tokens if self.file_max_tokens > 0 else 1.0
            
            if extract_remaining and utilization_rate < 0.8:
                # 將最後一部分視為零碎剩餘
                remaining_chunks.extend(current_chunks)
                print(f"文件 {source_file} 最後 {len(current_chunks)} 個 chunks (tokens: {current_tokens}, 利用率: {utilization_rate:.2%}) 被提取為剩餘部分")
            else:
                # 正常處理最後一部分
                file_key = f"{source_file}_merged_part{current_part}" if current_part > 1 else source_file
                
                complete_content = "\n\n".join([chunk["content"] for chunk in current_chunks])
                
                # 收集 ChromaDB ID
                chroma_ids = [chunk["chroma_id"] for chunk in current_chunks]
                
                # 更新 ChromaDB metadata 中的 group_id
                if update_metadata:
                    self._update_document_group_metadata(chroma_ids, file_key)
                
                result[file_key] = {
                    "content": complete_content,
                    "total_token_count": current_tokens,
                    "chroma_ids": chroma_ids,
                    "chunk_ids": [chunk["chunk_id"] for chunk in current_chunks]
                }
        
        return result, remaining_chunks
    
    def _merge_remaining_chunks_by_similarity(
        self, remaining_chunks: List[Dict[str, any]], 
        update_metadata: bool = True
    ) -> Dict[str, Dict[str, any]]:
        """
        將剩餘的零碎 chunks 按相似度進行分組合併
        
        參數:
            remaining_chunks: 剩餘的 chunks 列表
            update_metadata: 是否更新 ChromaDB 中的 metadata
            
        返回:
            按相似度分組合併後的文件字典
        """
        if not remaining_chunks:
            return {}
        
        print(f"開始對 {len(remaining_chunks)} 個剩餘 chunks 按相似度進行合併...")
        
        # 構建 embeddings 矩陣
        embeddings_list = []
        for chunk in remaining_chunks:
            embeddings_list.append(chunk["embeddings"])
        
        embeddings_array = np.array(embeddings_list)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        # 使用貪心算法進行分組
        result = {}
        used_indices = set()
        group_id = 0
        
        for i in range(len(remaining_chunks)):
            if i in used_indices:
                continue
            
            # 開始新分組
            current_group = [remaining_chunks[i]]
            current_tokens = remaining_chunks[i]["token_count"]
            used_indices.add(i)
            
            # 按相似度排序剩餘的 chunks
            similarities = similarity_matrix[i]
            candidate_indices = [
                (j, similarities[j]) for j in range(len(remaining_chunks))
                if j not in used_indices
            ]
            candidate_indices.sort(key=lambda x: x[1], reverse=True)
            
            # 嘗試添加相似 chunks 到當前分組
            for j, similarity in candidate_indices:
                if current_tokens + remaining_chunks[j]["token_count"] <= self.file_max_tokens:
                    current_group.append(remaining_chunks[j])
                    current_tokens += remaining_chunks[j]["token_count"]
                    used_indices.add(j)
            
            # 創建合併的文件
            file_key = f"remaining_chunks_group_{group_id}"
            
            merged_content = "\n\n=== REMAINING CHUNK ===\n\n".join([
                chunk["content"] for chunk in current_group
            ])
            
            chroma_ids = [chunk["chroma_id"] for chunk in current_group]
            chunk_ids = [chunk["chunk_id"] for chunk in current_group]
            
            # 計算平均相似度
            if len(current_group) > 1:
                similarities_in_group = []
                indices_in_group = []
                # 獲取當前分組中所有 chunk 的索引
                for chunk in current_group:
                    idx = [idx for idx, c in enumerate(remaining_chunks) if c["chroma_id"] == chunk["chroma_id"]][0]
                    indices_in_group.append(idx)
                
                # 計算分組內所有 chunk 對之間的相似度
                for p in range(len(indices_in_group)):
                    for q in range(p + 1, len(indices_in_group)):
                        similarities_in_group.append(similarity_matrix[indices_in_group[p], indices_in_group[q]])
                
                average_similarity = np.mean(similarities_in_group) if similarities_in_group else 1.0
            else:
                average_similarity = 1.0
            
            # 更新 ChromaDB metadata
            if update_metadata:
                self._update_document_group_metadata(chroma_ids, file_key)
            
            result[file_key] = {
                "content": merged_content,
                "total_token_count": current_tokens,
                "chroma_ids": chroma_ids,
                "chunk_ids": chunk_ids,
                "average_similarity": average_similarity,
                "group_type": "similarity_merged_remaining"
            }
            
            print(f"相似度分組合併: {file_key} - {len(current_group)} 個 chunks, tokens: {current_tokens}, 相似度: {average_similarity:.3f}")
            
            group_id += 1
        
        print(f"完成剩餘 chunks 的相似度合併，共產生 {len(result)} 個分組")
        
        return result

    def build_complete_files_with_splitting(
        self, file_content_dict: Dict[str, List[Dict[str, any]]],
        update_metadata: bool = False
    ) -> Dict[str, Dict[str, any]]:
        """
        透過按順序連接頁面內容來建立完整的文件內容
        如果文件超過 max_tokens，將被分割成多個部分
        
        參數:
            file_content_dict: 映射來源文件到頁面內容列表的字典
            update_metadata: 是否更新 ChromaDB 中的 metadata
        返回:
            映射來源文件名（可能帶有 _part 後綴）到完整文件內容與 token 計數的字典
        """
        file_whole_content: Dict[str, Dict[str, any]] = {}
        
        for source_file, content_list in file_content_dict.items():
            # 按頁面編號排序
            content_list.sort(key=lambda x: x["page_number"])
            
            # 如果超過最大 token 則分割文件
            split_files = self._split_large_file_into_parts(source_file, content_list, update_metadata)
            
            # 將所有分割的文件添加到結果中
            file_whole_content.update(split_files)
        
        return file_whole_content
    
    def _split_large_file_into_parts(
        self, source_file: str, content_list: List[Dict[str, any]], 
        update_metadata: bool = False
    ) -> Dict[str, Dict[str, any]]:
        """
        根據 token 限制將文件分割成多個部分
        
        參數:
            source_file: 原始來源文件名
            content_list: 頁面內容字典列表
            
        返回:
            包含分割文件部分的字典
        """
        result = {}
        current_part = 1
        current_content = []
        current_tokens = 0
        
        for page_data in content_list:
            page_tokens = page_data["token_count"]
            
            # 特殊情況：如果單個頁面就超過限制，記錄警告但仍然處理
            if page_tokens > self.file_max_tokens:
                from loguru import logger
                logger.warning(f"頁面 {page_data.get('page_number', 'unknown')} 的 token 數 ({page_tokens}) 超過限制 ({self.file_max_tokens})，但仍將其添加到分組中")
                # 注意：單個頁面超過限制時，我們仍然將其加入當前分組
                # 這是因為無法再進一步分割頁面內容
            
            # 如果添加此頁面將超過限制，則保存當前部分並開始新部分
            if current_tokens + page_tokens > self.file_max_tokens and current_content:
                # 保存當前部分
                file_key = f"{source_file}_merged_part{current_part}" if current_part > 1 or len(content_list) > len(current_content) else source_file
                
                complete_content = "\n\n\n".join([item["content"] for item in current_content])
                
                # 計算平均 embedding
                embeddings = [item["embeddings"] for item in current_content]
                embeddings_array = np.array(embeddings)
                mean_embedding = np.mean(embeddings_array, axis=0)
                
                # L2 正規化
                l2_norm = np.linalg.norm(mean_embedding)
                normalized_embedding = mean_embedding if l2_norm == 0 else mean_embedding / l2_norm
                
                # 收集 ChromaDB ID
                chroma_ids = [item["chroma_id"] for item in current_content]
                
                # 更新 ChromaDB metadata 中的 group_id
                if update_metadata:
                    self._update_document_group_metadata(chroma_ids, file_key)
                
                result[file_key] = {
                    "content": complete_content,
                    "total_token_count": current_tokens,
                    "normalized_embedding": normalized_embedding.tolist() if hasattr(normalized_embedding, 'tolist') else normalized_embedding,
                    "chroma_ids": chroma_ids
                }
                
                # 開始新部分
                current_part += 1
                current_content = []
                current_tokens = 0
            
            # 將當前頁面添加到當前部分
            current_content.append(page_data)
            current_tokens += page_tokens
        
        # 保存最後一部分
        if current_content:
            file_key = f"{source_file}_merged_part{current_part}" if current_part > 1 else source_file
            
            complete_content = "\n\n\n".join([item["content"] for item in current_content])
            
            # 計算平均 embedding
            embeddings = [item["embeddings"] for item in current_content]
            embeddings_array = np.array(embeddings)
            mean_embedding = np.mean(embeddings_array, axis=0)
            
            # L2 正規化
            l2_norm = np.linalg.norm(mean_embedding)
            normalized_embedding = mean_embedding if l2_norm == 0 else mean_embedding / l2_norm
            
            # 收集 ChromaDB ID
            chroma_ids = [item["chroma_id"] for item in current_content]
            
            # 更新 ChromaDB metadata 中的 group_id
            if update_metadata:
                self._update_document_group_metadata(chroma_ids, file_key)
            
            result[file_key] = {
                "content": complete_content,
                "total_token_count": current_tokens,
                "normalized_embedding": normalized_embedding.tolist() if hasattr(normalized_embedding, 'tolist') else normalized_embedding,
                "chroma_ids": chroma_ids
            }
        
        return result
    
    def _update_document_group_metadata(self, chroma_ids: List[str], group_id: str) -> None:
        """
        更新 ChromaDB metadata 以為指定文件添加 group_id
        
        參數:
            chroma_ids: 要更新的 ChromaDB 文件 ID 列表
            group_id: 要分配給這些文件的 group_id（文件鍵）
        """
        try:
            # 獲取這些文件的當前 metadata 和 documents（回傳順序可能與 chroma_ids 不同）
            current_data = self.chroma.get(ids=chroma_ids)

            if not current_data.get("metadatas"):
                print(f"未找到 ID 的 metadata: {chroma_ids}")
                return

            doc_contents = current_data.get("documents", [])
            doc_metadatas = current_data.get("metadatas", [])
            returned_ids = current_data.get("ids", [])

            # 建立映射，以便依傳入的 chroma_ids 順序重建 documents
            # 優先以回傳的 ids 對齊；若無，使用 metadata.chunk_id（與 chroma_id 相同）
            by_id_content = {}
            by_id_metadata = {}
            by_chunkid_content = {}
            by_chunkid_metadata = {}

            for idx in range(min(len(doc_contents), len(doc_metadatas))):
                content = doc_contents[idx]
                metadata = doc_metadatas[idx] or {}
                ret_id = returned_ids[idx] if idx < len(returned_ids) else None
                chunk_id = metadata.get("chunk_id")

                if ret_id is not None:
                    by_id_content[ret_id] = content
                    by_id_metadata[ret_id] = metadata
                if chunk_id is not None:
                    by_chunkid_content[chunk_id] = content
                    by_chunkid_metadata[chunk_id] = metadata

            # 依 chroma_ids 的順序建立 updated_documents，確保順序一致
            updated_documents = []
            for cid in chroma_ids:
                # 先用回傳 ids 對應
                content = by_id_content.get(cid)
                metadata = by_id_metadata.get(cid)
                if content is None or metadata is None:
                    # 回退用 chunk_id（與 chroma_id 相同）對應
                    content = by_chunkid_content.get(cid)
                    metadata = by_chunkid_metadata.get(cid, {})
                if metadata is None:
                    metadata = {}
                # 更新 metadata 添加 group_id
                metadata["group_id"] = group_id
                # 創建 Document 對象（即便 content 為 None，也保持佔位，避免順序錯亂）
                doc = Document(page_content=content if content is not None else "", metadata=metadata)
                updated_documents.append(doc)

            # 更新 ChromaDB，ids 與 documents 一一對齊
            self.chroma.update_documents(
                ids=chroma_ids,
                documents=updated_documents
            )

            print(f"已更新 {len(chroma_ids)} 個文件的 group_id: {group_id}")

        except Exception as e:
            print(f"更新 ChromaDB 中的 group_id 時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    def process_all_documents(self, merge_remaining_by_similarity: bool = True, save_folder_path: str = None) -> Dict[str, Dict[str, any]]:
        """
        檢索和處理 ChromaDB 內容的主要方法
        現在支持提取剩餘的零碎 chunks 並按相似度合併，並可選擇性保存結果
        
        參數:
            merge_remaining_by_similarity: 是否對剩餘的零碎 chunks 按相似度合併
            save_folder_path: 保存路徑（可選），如果為 None 則不保存
        
        返回:
            包含按來源文件組織的完整文件內容與 token 計數的字典
        """
        # 使用新的基於 chunk 的方法
        file_content_dict = self.get_chunk_content_from_chroma()
        file_whole_content, remaining_chunks = self.build_complete_files_with_chunk_splitting(
            file_content_dict,
            update_metadata =True, 
            extract_remaining=True
        )
        
        # 如果啟用相似度合併，處理剩餘的 chunks
        if merge_remaining_by_similarity and remaining_chunks:
            print(f"發現 {len(remaining_chunks)} 個剩餘的零碎 chunks，開始按相似度合併...")
            merged_remaining = self._merge_remaining_chunks_by_similarity(remaining_chunks)
            file_whole_content.update(merged_remaining)
        
        # 如果提供了保存路徑，保存處理結果
        if save_folder_path is not None:
            print(f"保存處理結果到: {save_folder_path}")
            self._save_expanded_result(file_whole_content, save_folder_path)
        
        return file_whole_content

    def prepare_kv_cache_content(self, save_folder_path: str = None) -> Dict[str, Dict[str, any]]:
        file_whole_content = self.process_all_documents(save_folder_path=save_folder_path)
        return file_whole_content
    
    def group_and_save_similar_documents(self, save_folder_path: str = None) -> Tuple[Dict[str, Dict[str, any]], np.ndarray, List[str], Dict[str, Dict[str, any]], Dict[str, List[Dict[str, any]]]]:
        """
        獲取按相似度擴展後的內容分組，並可選擇性保存結果
        
        參數:
            save_folder_path: 保存路徑（可選），如果為 None 則不保存
        
        返回:
            按相似度分組並擴展後的內容字典、相似度矩陣、文件名列表、原始文件內容字典、更新後的 file_content_dict 的元組
        """
        # 先獲取基本的文件內容
        content = self.process_all_documents()
        
        # 按 part 順序分組，並收集剩餘部分
        expanded_content, similarity_matrix, filenames, remaining_parts = self._group_documents_by_parts(content)
        
        # 如果有剩餘部分，按相似度重新分組
        if remaining_parts:
            print(f"發現 {len(remaining_parts)} 個剩餘部分，開始按相似度重新分組...")
            remaining_groups, remaining_similarity_matrix, remaining_filenames, file_content_dict = self._group_remaining_parts_by_similarity(remaining_parts, file_content_dict)
            
            # 合併順序分組和相似度分組的結果
            expanded_content.update(remaining_groups)
            filenames.extend(remaining_filenames)
            
            # 記錄分組統計
            sequential_groups = len(expanded_content) - len(remaining_groups)
            similarity_groups = len(remaining_groups)
            print(f"總分組數: {len(expanded_content)} (順序分組: {sequential_groups}, 相似度分組: {similarity_groups})")
        
        # 保存擴展後的內容（如果提供了路徑）
        if save_folder_path is not None:
            self._save_expanded_result(expanded_content, save_folder_path, content)
            
            # 同時儲存 chunks.json
            chunks_json_path = os.path.join(save_folder_path, "chunks.json")
            self.save_file_content_dict_as_chunks_json(file_content_dict, chunks_json_path)
        
        return expanded_content, similarity_matrix, filenames, content, file_content_dict
    
    def save_file_content_dict_as_chunks_json(self, file_content_dict: Dict[str, List[Dict[str, any]]], 
                                            save_path: str) -> None:
        """
        將 file_content_dict 儲存為 chunks.json 格式
        
        參數:
            file_content_dict: 包含每個 chunk 的詳細資訊的字典
            save_path: 儲存路徑（包含檔案名）
        """
        chunks_data = []
        
        for source_file, chunks in file_content_dict.items():
            for chunk in chunks:
                # 建立符合 chunks.json 格式的資料結構
                chunk_data = {
                    "page_content": chunk["content"],
                    "metadata": {
                        "source_file": source_file,
                        "source": chunk.get("metadata", {}).get("source", source_file),
                        "page": chunk.get("metadata", {}).get("page", "1_1"),
                        "parent_content": chunk["content"],
                        "page_key": chunk.get("metadata", {}).get("page_key", f"{source_file}-{chunk['chunk_index']}"),
                        "start_page": chunk.get("metadata", {}).get("start_page", 1),
                        "end_page": chunk.get("metadata", {}).get("end_page", 1),
                        "chunk_id": chunk["chunk_id"],
                        "chunk_index": chunk["chunk_index"],
                        "token_count": chunk["token_count"],
                        "group_id": chunk.get("metadata", {}).get("group_id", ""),
                    }
                }
                
                chunks_data.append(chunk_data)
        
        # 儲存為 JSON 檔案
        import json
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=4)
        
        print(f"已儲存 chunks.json: {save_path}")
        print(f"總共儲存了 {len(chunks_data)} 個 chunks")
    
    def save_similarity_matrix(self, similarity_matrix: np.ndarray, filenames: List[str], save_folder_path: str, file_token_counts: Dict[str, int] = None) -> None:
        """
        保存相似度矩陣到 Excel 文件
        
        參數:
            similarity_matrix: 相似度矩陣
            filenames: 文件名列表
            save_folder_path: 保存路徑
            file_token_counts: 每個文件的 token 數量字典（可選）
        """
        if similarity_matrix.size == 0 or not filenames:
            print("相似度矩陣為空，跳過保存")
            return
            
        # 確保資料夾存在並設置權限
        self._ensure_directory_permissions(save_folder_path)
        
        # 創建包含 token 資訊的文件名標籤
        if file_token_counts:
            enhanced_filenames = []
            for filename in filenames:
                token_count = file_token_counts.get(filename, 0)
                enhanced_filename = f"{filename} ({token_count} tokens)"
                enhanced_filenames.append(enhanced_filename)
            
            # 使用增強的文件名創建 DataFrame
            df = pd.DataFrame(similarity_matrix, index=enhanced_filenames, columns=enhanced_filenames)
        else:
            # 如果沒有 token 資訊，使用原始文件名
            df = pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)
        
        # 保存為 Excel 文件
        excel_filepath = os.path.join(save_folder_path, "similarity_matrix.xlsx")
        
        try:
            with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
                # 保存包含 token 資訊的矩陣
                df.to_excel(writer, sheet_name='Similarity_Matrix')
                
                # 保存統計資訊
                stats_data = {
                    '統計項目': ['最大相似度', '最小相似度', '平均相似度', '標準差', '文件數量'],
                    '數值': [
                        float(np.max(similarity_matrix)),
                        float(np.min(similarity_matrix[similarity_matrix != 1.0])),  # 排除自身相似度1.0
                        float(np.mean(similarity_matrix[similarity_matrix != 1.0])),
                        float(np.std(similarity_matrix[similarity_matrix != 1.0])),
                        len(filenames)
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                # 保存文件 token 統計資訊（如果有提供）- 作為詳細統計表
                if file_token_counts:
                    token_data = {
                        '文件名': list(file_token_counts.keys()),
                        'Token 數量': list(file_token_counts.values())
                    }
                    token_df = pd.DataFrame(token_data)
                    # 保持原來的順序，不按 Token 數量排序
                    token_df.to_excel(writer, sheet_name='File_Tokens_Detail', index=False)
                    
                    # 添加 Token 統計摘要到同一個工作表
                    token_values = list(file_token_counts.values())
                    if token_values:  # 確保有數據
                        token_summary = {
                            '統計項目': ['總 Token 數', '平均 Token 數', '最大 Token 數', '最小 Token 數', '標準差'],
                            '數值': [
                                sum(token_values),
                                round(np.mean(token_values), 2),
                                max(token_values),
                                min(token_values),
                                round(np.std(token_values), 2)
                            ]
                        }
                        token_summary_df = pd.DataFrame(token_summary)
                        
                        # 在同一個工作表中添加統計資訊，空幾行後添加
                        startrow = len(token_df) + 3
                        token_summary_df.to_excel(writer, sheet_name='File_Tokens_Detail', 
                                                startrow=startrow, index=False)
                
            # 設置文件權限
            self._ensure_file_permissions(excel_filepath)
            print(f"已保存相似度矩陣到: {excel_filepath}")
            print(f"矩陣大小: {similarity_matrix.shape[0]} x {similarity_matrix.shape[1]}")
            if file_token_counts:
                print(f"已包含 {len(file_token_counts)} 個文件的 Token 統計資訊")
            
        except Exception as e:
            print(f"保存相似度矩陣時發生錯誤: {e}")
            # 嘗試保存為 CSV 作為後備選項
            try:
                csv_filepath = os.path.join(save_folder_path, "similarity_matrix.csv")
                df.to_csv(csv_filepath, encoding='utf-8-sig')
                self._ensure_file_permissions(csv_filepath)
                print(f"已保存相似度矩陣為 CSV 格式: {csv_filepath}")
            except Exception as csv_error:
                print(f"保存 CSV 格式也失敗了: {csv_error}")
  
    def _save_expanded_result(self, expanded_content: Dict[str, Dict[str, any]], save_folder_path: str) -> None:
        """
        保存處理後的內容到文件系統
        
        參數:
            expanded_content: 處理後的內容字典，包含 content, total_token_count, chroma_ids 等
            save_folder_path: 保存資料夾路徑
        """
        # 確保資料夾存在並設置權限
        self._ensure_directory_permissions(save_folder_path)
        
        # 確保 merged_files 目錄存在
        # merged_files_dir = os.path.join(save_folder_path, "merged_files")
        # self._ensure_directory_permissions(merged_files_dir)
        
        # 準備統計資料字典（不包含 content）
        statistics_data = {}
        total_files_saved = 0
        
        for file_key, file_data in expanded_content.items():
            # 複製字典但排除 content 欄位
            stats = {k: v for k, v in file_data.items() if k != "content"}
            statistics_data[file_key] = stats
            
            # 保存合併後的內容到 merged_files 目錄
            if "content" in file_data:
                txt_filename = f"{file_key}.txt"
                merged_filepath = os.path.join(save_folder_path, txt_filename)
                
                try:
                    with open(merged_filepath, 'w', encoding='utf-8') as f:
                        f.write(file_data["content"])
                    self._ensure_file_permissions(merged_filepath)
                    print(f"已保存合併文件: {merged_filepath}")
                    total_files_saved += 1
                except Exception as e:
                    print(f"保存文件失敗 {merged_filepath}: {e}")
        
        # 保存統計資料為 JSON
        json_filepath = os.path.join(save_folder_path, "statistics.json")
        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, ensure_ascii=False, indent=2)
            self._ensure_file_permissions(json_filepath)
            print(f"已保存統計資料: {json_filepath}")
        except Exception as e:
            print(f"保存統計資料失敗: {e}")
        
        print(f"總共處理了 {len(expanded_content)} 個文件，保存了 {total_files_saved} 個合併文件")
        

class SimpleFakeEmbeddings:
    """簡單的假 embedding 類別，用於測試（與建立時相同）"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.rng = np.random.default_rng(42)  # 使用固定種子確保可重現性
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """為多個文檔生成 embedding"""
        vectors = []
        for text in texts:
            length = max(1, len(text))
            rng_local = np.random.default_rng(length)
            vec = rng_local.normal(size=self.dimension)
            # L2 正規化
            norm = (vec**2).sum() ** 0.5
            if norm != 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors
    
    def embed_query(self, text: str) -> List[float]:
        """為單個查詢生成 embedding"""
        return self.embed_documents([text])[0]

if __name__ == "__main__":
    persist_dir = r"D:\AI\Code\agentbuilder\km-for-agent-builder\tmp\km-docproc-test-20251028_100520\test_collection_20251028_100520\chroma_db"
    collection_name = "test_collection_20251028_100520"
    save_folder_path = r"D:\AI\Code\agentbuilder\km-for-agent-builder\tmp\km-docproc-test-20251028_100520"
    embeddings = SimpleFakeEmbeddings(dimension=64)
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=collection_name
    )
                
    handler = KVcacheContentHandler(vectorstore)
    content = handler.process_all_documents(save_folder_path=save_folder_path)



    

    
    

