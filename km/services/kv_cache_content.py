"""
ChromaDB 內容檢索與處理模組

此模組提供 KVcacheContentHandler 類別，用於連接 ChromaDB 實例，
檢索文件內容，並按來源文件進行組織和適當的頁面排序。
"""
import re
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

import numpy as np
import requests
from config import settings

# 配置 loguru logger（如果尚未配置）
# 檢查是否已經有 handler，如果沒有則添加默認的終端輸出
# 注意：不添加檔案 handler，避免產生多個 log 檔（由 api.py 統一管理）
try:
    # 嘗試獲取 logger 的 handlers 數量
    handlers_count = len(logger._core.handlers)
    if handlers_count == 0:
        # 如果沒有 handler，只添加終端輸出（不添加檔案 handler）
        logger.add(
            sys.stdout,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{function}:{line} - {message}",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
except (AttributeError, TypeError):
    # 如果無法檢查 handlers，跳過配置（由 api.py 統一管理）
    # 避免在無法檢查時重複添加 handler
    pass
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document

# from services.lib.tokenizer_manager import TokenCounter
# token_counter = TokenCounter(tokenizer_path=r"D:\AI\Code\agentbuilder\km-for-agent-builder\Mistral-Small-24B-Instruct-2501_tokenizer.json")


def count_tokens(text: str) -> int:
    """
    計算文本的 token 數量
    
    參數:
        text: 要計算 token 數的文本
        
    返回:
        token 數量
    """
    base_url = re.sub(r'/v\d+$', '', settings.LLM_URL)
    API_URL = f"{base_url}/tokenize"
    MODEL = settings.LLM_MODEL_NAME
    CONTENT = text
    if settings.LLM_TYPE == "llamacpp":
        # 發送請求
        payload = {
            "content": CONTENT
        }
    elif settings.LLM_TYPE == "vllm":
        payload = {
            "model": MODEL,
            "prompt": CONTENT
        }
    # logger.info(f"Count tokens: API_URL {API_URL} {MODEL} {payload.keys()}")
    response = requests.post(API_URL, json=payload, timeout=30)
    result = response.json()
    count_tokens = len(result['tokens'])
    # count_tokens = token_counter(text) 
    logger.info(f"Count tokens: {count_tokens}")
    
    return count_tokens


class KVcacheContentHandler:
    """
    ChromaDB 內容檢索與處理類別
    
    此類別處理從 ChromaDB 實例檢索文件內容，
    並按來源文件進行組織和適當的頁面排序。
    """
    
    def __init__(self, 
                 chroma: Chroma,
                 file_max_tokens: int = 16000):
        """
        初始化 KVcacheContentHandler
        
        參數:
            chroma: Chroma 向量存儲實例
            file_max_tokens: 每個文件的最大 token 數，超過將被分割
        """
        self.chroma = chroma
        self.file_max_tokens = file_max_tokens
    
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
            logger.error(f"設置目錄權限失敗: {directory_path}, 錯誤: {e}")
    
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
            logger.error(f"設置文件權限失敗: {file_path}, 錯誤: {e}")
    
    def _extract_chunk_ids(self, chunks: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        從 chunks 列表中提取 chroma_ids 和 chunk_ids
        
        參數:
            chunks: chunk 字典列表
            
        返回:
            tuple: (chroma_ids, chunk_ids) - 兩個 ID 列表
        """
        chroma_ids = [chunk.get('chroma_id') for chunk in chunks if chunk.get('chroma_id')]
        chunk_ids = [chunk.get('chunk_id') for chunk in chunks if chunk.get('chunk_id')]
        return chroma_ids, chunk_ids

    def _normalize_filename(self, filename: str) -> str:
        """
        標準化檔名，將副檔名轉換為底線連接形式
        例如: "example.txt" -> "example_txt"
             "document.pdf" -> "document_pdf"
             "no_extension" -> "no_extension"
        
        參數:
            filename: 原始檔名
            
        返回:
            標準化後的檔名（不含點號）
        """
        if "." in filename:
            name_parts = filename.rsplit(".", 1)
            return f"{name_parts[0]}_{name_parts[1]}"
        return filename

    def _calc_similarity_by_chunks_for_two_files(self, 
                                                 file_a: Tuple[str, List[Dict[str, Any]], int], 
                                                 file_b: Tuple[str, List[Dict[str, Any]], int]) -> float:
        """
        計算兩個文件之間的相似度
        
        參數:
            file_a: 文件 A，格式為 (source_file, content_list, file_total_token)
            file_b: 文件 B，格式為 (source_file, content_list, file_total_token)
            
        返回:
            兩個文件之間的平均相似度分數（0-1 之間）
        """
        content_list_a = file_a[1]
        content_list_b = file_b[1]

        scores = []
        for chunk_a in content_list_a:
            emb_a = chunk_a.get('embeddings', [])
            embs_b = [chunk_b.get('embeddings', []) for chunk_b in content_list_b]

            similarity_matrix = cosine_similarity([emb_a], embs_b)
            best_score = np.max(similarity_matrix)
            scores.append(best_score)

        average_score = np.mean(scores)

        return average_score

    def get_chunk_content_from_chroma(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        從 ChromaDB 檢索 chunk 內容並按來源文件和 chunk_index 組織
        
        返回:
            映射來源文件名到包含 chunk 索引、內容和 token 數量的內容字典列表
        """
        data = self.chroma.get(include=["embeddings", "metadatas", "documents"])
        
        file_content_dict: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for idx in range(len(data["metadatas"])):
            metadata = data["metadatas"][idx]
            source_file = metadata["source"]
            chunk_content = data["documents"][idx] if "documents" in data and idx < len(data["documents"]) else ""
            
            # 獲取 chunk_index，如果沒有則設為 0
            chunk_index = metadata.get("chunk_index", 0)
            chunk_id = metadata.get("chunk_id", f"{source_file}_part{chunk_index}")
            chroma_id = data["ids"][idx]
            
            # 如果有 tokenizer 則計算 token 數量
            token_count = 0
            # logger.info(f"Chunk content: {chunk_content}")
            token_count = count_tokens(chunk_content)


            # 建立內容字典
            # 注意：chroma_id 與 chunk_id 應該相同，但保留 chroma_id 以確保向後兼容性
            # 如果確認所有文檔都使用 chunk_id 作為 ChromaDB ID，可以移除 chroma_id
            content_dict = {
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
                "content": chunk_content,
                "token_count": token_count,
                "embeddings": data["embeddings"][idx],
                "chroma_id": chroma_id,  # 與 chunk_id 相同，保留以確保向後兼容
                "metadata": metadata
            }
            
            file_content_dict[source_file].append(content_dict)
        
        # 對每個文件的 chunks 按 chunk_index 排序
        for source_file in file_content_dict:
            file_content_dict[source_file].sort(key=lambda x: x["chunk_index"])
        
        return file_content_dict
    
    def _classify_file_by_size(self, 
                               source_file: str, 
                               content_list: List[Dict[str, Any]], 
                               max_token_per_group: int) -> Tuple[bool, int]:
        """
        根據文件大小分類為大檔案或小檔案
        
        參數:
            source_file: 來源文件名
            content_list: chunk 列表
            max_token_per_group: 每個分組的最大 token 數
            
        返回:
            tuple: (is_large_file, file_total_token_accurate)
                - is_large_file: 是否為大檔案
                - file_total_token_accurate: 移除重疊後的精確 token 數
        """
        # 排序 content_list by chunk_index
        content_list.sort(key=lambda x: x["chunk_index"])
        
        # 預處理：計算每個 chunk 移除重疊後的增量 token 數
        content_list, file_total_token_accurate = self._preprocess_chunks_with_overlap_removal(content_list)
        
        # 原始的累積 token 數（未移除重疊）
        file_total_token_raw = sum(item.get('token_count', 0) or 0 for item in content_list)
        
        logger.info(f'檔案: {source_file}')
        logger.info(f'  - 原始累積 token: {file_total_token_raw}')
        logger.info(f'  - 移除重疊後 token: {file_total_token_accurate} (精確值)')
        logger.info(f'  - Max token per group: {max_token_per_group}')
        
        is_large_file = file_total_token_accurate >= max_token_per_group
        logger.info(f'  - 判斷為: {"大檔案" if is_large_file else "小檔案"}')
        
        return is_large_file, file_total_token_accurate
    
    def build_merged_files_with_chunk_splitting(self, 
                                                file_content_dict: Dict[str, List[Dict[str, Any]]], 
                                                update_metadata: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        構建合併文件，根據文件大小進行分割或合併
        
        參數:
            file_content_dict: 按來源文件組織的 chunk 內容字典
            update_metadata: 是否更新 ChromaDB metadata
            
        返回:
            合併後的文件內容字典，key 為文件鍵，value 包含 content、total_token_count、chunk_ids 等
        """
        file_whole_content: Dict[str, Dict[str, Any]] = {}
        max_token_per_group = settings.MAX_TOKENS_PER_GROUP
        small_files = []
        
        for source_file, content_list in file_content_dict.items():
            # 分類文件大小
            is_large_file, file_total_token_accurate = self._classify_file_by_size(
                source_file, content_list, max_token_per_group
            )
            
            if is_large_file:
                # 處理大檔案：分割成多個 part
                large_merged_parts = self._split_large_file_into_merged_parts(
                    source_file, 
                    content_list, 
                    file_total_token_accurate,
                    update_metadata
                )
                file_whole_content.update(large_merged_parts)
            else:
                # 收集小檔案，稍後按相似度合併
                small_files.append((source_file, content_list, file_total_token_accurate))

        # 處理小檔案：按相似度合併
        if small_files:
            small_files.sort(key=lambda x: x[2], reverse=True)
            small_merged_parts = self._combine_small_files_by_similarity_into_merged_parts(
                small_files, update_metadata
            )
            file_whole_content.update(small_merged_parts)

        return file_whole_content

    def _find_overlap(self, prev_text: str, current_text: str, max_overlap_chars: int = 2000) -> int:
        """
        找到前一個文本的結尾與當前文本開頭之間的重疊長度
        
        參數:
            prev_text: 前一個 chunk 的文本
            current_text: 當前 chunk 的文本
            max_overlap_chars: 最大重疊字符數（用於限制搜索範圍）
            
        返回:
            重疊的字符數（從 prev_text 結尾開始計算）
        """
        if not prev_text or not current_text:
            return 0
        
        # 限制搜索範圍以提高效率
        search_length = min(len(prev_text), len(current_text), max_overlap_chars)
        prev_suffix = prev_text[-search_length:]
        current_prefix = current_text[:search_length]
        
        # 從最長可能的重疊開始，逐步減少長度
        for overlap_len in range(search_length, 0, -1):
            if prev_suffix[-overlap_len:] == current_prefix[:overlap_len]:
                return overlap_len
        
        return 0

    def _extract_chunk_without_overlap(
        self,
        last_chunk_content: Optional[str],
        chunk_content: str,
    ) -> Tuple[str, int]:
        """
        從 chunk_content 中移除與上一個 chunk 的重疊部分，並返回新增內容與重疊長度
        
        參數:
            last_chunk_content: 上一個 chunk 的內容（可為 None）
            chunk_content: 當前 chunk 的內容
            
        返回:
            tuple: (new_segment, overlap_len)
                - new_segment: 移除重疊後的新增內容片段
                - overlap_len: 重疊的字符長度
        """
        if not chunk_content:
            return '', 0

        overlap_len = self._find_overlap(last_chunk_content or '', chunk_content)
        new_segment = chunk_content[overlap_len:] if overlap_len > 0 else chunk_content
        return new_segment, overlap_len

    def _preprocess_chunks_with_overlap_removal(self, content_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """
        預處理 content_list，為每個 chunk 計算移除重疊後的增量 token 數和內容片段
        
        參數:
            content_list: 已排序的 chunk 列表
            
        返回:
            tuple: (處理後的 content_list, 實際總 token 數)
                - 每個 chunk 會新增：
                  * 'token_count_without_overlap': 移除重疊後的增量 token 數
                  * 'new_segment_without_overlap': 移除重疊後的新增內容片段（用於後續合併）
                - 實際總 token 數是移除所有重疊後的精確值
        """
        if not content_list:
            return content_list, 0
        
        total_token_without_overlap = 0
        last_chunk_content = None
        
        for idx, chunk in enumerate(content_list):
            chunk_content = chunk.get('content', '') or ''
            
            if not chunk_content:
                chunk['token_count_without_overlap'] = 0
                chunk['new_segment_without_overlap'] = ''
                continue
            
            if idx == 0:
                # 第一個 chunk，沒有重疊
                token_count = count_tokens(chunk_content)
                # logger.info(f"Chunk content: {chunk_content}, token count: {token_count}")
                chunk['token_count_without_overlap'] = token_count
                chunk['new_segment_without_overlap'] = chunk_content  # 第一個 chunk 的完整內容
                total_token_without_overlap += token_count
                last_chunk_content = chunk_content
                logger.info(f'預處理 chunk {idx}: 第一個 chunk，token 數 = {token_count}')
            else:
                # 第二個以後的 chunk，檢查重疊
                new_segment, overlap_len = self._extract_chunk_without_overlap(last_chunk_content, chunk_content)
                
                # 保存 new_segment 到 chunk 中，避免後續重複計算
                chunk['new_segment_without_overlap'] = new_segment
                
                if new_segment:
                    # 計算新增內容的 token 數（不包含分隔符，因為這裡只是計算實際內容）
                    incremental_token_count = count_tokens(new_segment)
                    chunk['token_count_without_overlap'] = incremental_token_count
                    total_token_without_overlap += incremental_token_count
                    
                    if overlap_len > 0:
                        logger.info(f'預處理 chunk {idx}: 發現重疊 {overlap_len} 字元，移除後 token 數 = {incremental_token_count} (原始: {chunk.get("token_count", 0)})')
                    else:
                        logger.info(f'預處理 chunk {idx}: 無重疊，token 數 = {incremental_token_count}')
                else:
                    # 完全重疊
                    chunk['token_count_without_overlap'] = 0
                    logger.info(f'預處理 chunk {idx}: 完全重疊，token 數 = 0 (原始: {chunk.get("token_count", 0)})')
                
                last_chunk_content = chunk_content
        
        logger.info(f'預處理完成: 原始累積 token = {sum(c.get("token_count", 0) or 0 for c in content_list)}, 移除重疊後 token = {total_token_without_overlap}')
        return content_list, total_token_without_overlap
    
    def _get_chunk_incremental_info(
        self,
        last_chunk_content: Optional[str],
        chunk: Dict[str, Any],
    ) -> Tuple[str, int]:
        """
        計算新 chunk 移除重疊後的增量資訊（用於增量計算 token 數）
        優先使用預處理時計算好的值，完全避免重複計算
        
        參數:
            last_chunk_content: 上一個 chunk 的內容（不包含 overlap，僅用於 fallback 情況）
            chunk: 當前要處理的 chunk
            
        返回:
            tuple: (new_segment, incremental_token_count)
                - new_segment: 移除重疊後的新增內容片段
                - incremental_token_count: 新增的 token 數（不包含分隔符）
        """
        # 優先使用預處理時計算好的值（完全避免重複計算）
        if 'token_count_without_overlap' in chunk and 'new_segment_without_overlap' in chunk:
            new_segment = chunk['new_segment_without_overlap']
            incremental_token_count = chunk['token_count_without_overlap']
            return new_segment, incremental_token_count
        
        # 如果沒有預處理過，則即時計算（fallback，向後兼容）
        chunk_content = chunk.get('content', '') or ''
        if not chunk_content:
            return '', 0
        
        new_segment, _ = self._extract_chunk_without_overlap(last_chunk_content, chunk_content)

        if new_segment:
            # 計算新增內容的 token 數（不包含分隔符）
            incremental_token_count = count_tokens(new_segment)
            return new_segment, incremental_token_count
        else:
            # 完全重疊，沒有新增內容
            return '', 0

    def _split_large_file_into_merged_parts(self, 
                                            source_file: str, 
                                            content_list: List[Dict[str, Any]], 
                                            file_total_token: int, 
                                            update_metadata: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        將大檔案分割成多個合併部分
        
        參數:
            source_file: 來源文件名
            content_list: 已預處理的 chunk 列表（包含 token_count_without_overlap 等）
            file_total_token: 文件的總 token 數（移除重疊後）
            update_metadata: 是否更新 ChromaDB metadata
            
        返回:
            合併部分字典，key 為文件鍵（如 filename_merged_part1），value 包含 content、total_token_count、chunk_ids
        """
        logger.info(f'split large file: {source_file}')
        # 標準化檔名，移除副檔名
        normalized_filename = self._normalize_filename(source_file)
        max_token_per_group = settings.MAX_TOKENS_PER_GROUP
        merged_content = ''
        merged_parts = {}  
        current_chunks = [] 
        current_token_count = 0
        last_chunk_content = None
        part_counter = 1
        # 保存每個部分的完整資訊，用於後續調整
        saved_parts = []

        # 直接使用 max_token_per_group 作為切分標準
        token_per_group = max_token_per_group
        logger.info(f'token_per_group: {token_per_group}')

        for idx, chunk in enumerate(content_list):
            # 直接讀取預處理後的增量資訊（無 overlap）
            new_segment = chunk.get('new_segment_without_overlap', '') or ''
            incremental_token_count = chunk.get('token_count_without_overlap')

            # 若尚未預處理到，fallback 即時計算
            if incremental_token_count is None:
                new_segment, incremental_token_count = self._get_chunk_incremental_info(
                    last_chunk_content,
                    chunk
                )

            # 若沒有新增內容則略過，但仍維持 last_chunk_content 與 current_chunks
            if not new_segment or incremental_token_count == 0:
                last_chunk_content = new_segment
                current_chunks.append(chunk)
                logger.info('chunk 無新增內容或完全重疊，略過累加')
                continue

            # 如果目前沒有累積的 chunks，視為新部分開頭
            if not current_chunks:
                chunk_content = chunk.get('content', '')
                chunk_token_count = chunk.get('token_count', 0)
                merged_content = chunk_content
                current_token_count = chunk_token_count
                current_chunks.append(chunk)
                logger.info(f'開始新部分，第一個 chunk token 數: {current_token_count}')
                continue

            # 計算添加這個 chunk 後的總 token 數
            predicted_token_count = current_token_count + incremental_token_count
            
            # 檢查是否會超過限制
            if predicted_token_count > token_per_group:
                # 會超過限制，不添加當前 chunk，先保存目前累積的部分
                logger.info(f'預測添加 chunk 後會超過限制 ({predicted_token_count} > {token_per_group})，不添加此 chunk')
                logger.info(f'保存部分: 包含 {len(current_chunks)} 個 chunks，原始累積 token (估算): {sum(c.get("token_count", 0) or 0 for c in current_chunks)}，實際 token (移除重疊後): {current_token_count}')
                
                # 安全檢查：確保當前部分不超過限制
                if current_token_count > token_per_group:
                    logger.warning(f'警告：當前部分 token 數 ({current_token_count}) 超過限制 ({token_per_group})！')
                
                # Generate file_key when saving the part
                file_key = f"{normalized_filename}_merged_part{part_counter}"
                part_counter += 1
                
                # 保存合併部分
                self._save_merged_part(
                    file_key=file_key,
                    merged_content=merged_content,
                    current_chunks=current_chunks,
                    current_token_count=current_token_count,
                    merged_parts=merged_parts,
                    saved_parts=saved_parts,
                    update_metadata=update_metadata
                )

                # 重置狀態，導致超過限制的這個 chunk 成為新部分的第一個 chunk
                # 第一部份使用完整的 chunk 內容（包含 overlap），保持完整性
                chunk_content = chunk.get('content', '')
                chunk_token_count = chunk.get('token_count', 0)
                merged_content = chunk_content
                current_token_count = chunk_token_count
                last_chunk_content = chunk_content
                current_chunks = [chunk]
                logger.info(f'已保存部分 {part_counter-1}，開始新部分 {part_counter}，第一個 chunk token 數: {current_token_count}')
            else:
                # 不超過限制，使用增量累加
                merged_content = f"{merged_content}{new_segment}"
                current_token_count = predicted_token_count
                last_chunk_content = new_segment
                current_chunks.append(chunk)
                logger.info(f'累加 chunk，增量 token: {incremental_token_count}，當前總 token: {current_token_count}')

        # 處理最後一個部分（循環結束後剩餘的內容）
        if current_chunks:
            last_part_token = current_token_count
            logger.info(f'處理最後一個部分，當前 token 數: {last_part_token} / {max_token_per_group} ({last_part_token/max_token_per_group*100:.1f}%)')
            
            # 嘗試平衡最後一部分的 token 數
            current_chunks, merged_content, last_part_token = self._try_balance_last_part(
                current_chunks=current_chunks,
                merged_content=merged_content,
                last_part_token=last_part_token,
                saved_parts=saved_parts,
                max_token_per_group=max_token_per_group
            )
            
            # Generate file_key when saving the last part
            file_key = f"{normalized_filename}_merged_part{part_counter}"
            
            # 最終安全檢查
            if last_part_token > max_token_per_group:
                logger.warning(f'警告：最後一部分 token 數 ({last_part_token}) 超過限制 ({max_token_per_group})！')
            
            logger.info(f'保存最後一部分 {part_counter}，包含 {len(current_chunks)} 個 chunks，token 數: {last_part_token}')
            
            # 保存最後一部分
            chroma_ids, chunk_ids = self._extract_chunk_ids(current_chunks)
            if update_metadata:
                self._update_document_group_metadata(chroma_ids, file_key)
            
            merged_parts[file_key] = {
                "content": merged_content,
                "total_token_count": last_part_token,
                "chunk_ids": chunk_ids
            }

        # 輸出最終統計資訊
        logger.info(f'===== 檔案切分完成 =====')
        logger.info(f'檔案: {source_file} (標準化為: {normalized_filename})')
        logger.info(f'總共切分為 {len(merged_parts)} 個部分')
        for part_key, part_info in merged_parts.items():
            token_count = part_info['total_token_count']
            utilization = token_count / max_token_per_group * 100
            logger.info(f'  - {part_key}: {token_count} tokens ({utilization:.1f}% 利用率)')
        logger.info(f'=======================')

        return merged_parts

    def _get_file_content_without_overlap(self, content_list: List[Dict[str, Any]]) -> str:
        """
        從 content_list 中提取所有不包含重疊的內容片段並合併
        
        參數:
            content_list: chunk 字典列表
            
        返回:
            合併後的完整內容字符串
        """
        file_content = ''
        for chunk in content_list:
            file_content += chunk.get('new_segment_without_overlap', '')
        return file_content
    
    def _save_merged_part(self,
                         file_key: str,
                         merged_content: str,
                         current_chunks: List[Dict[str, Any]],
                         current_token_count: int,
                         merged_parts: Dict[str, Dict[str, Any]],
                         saved_parts: List[Dict[str, Any]],
                         update_metadata: bool = False) -> None:
        """
        保存一個合併部分到 merged_parts 和 saved_parts
        
        參數:
            file_key: 文件鍵名
            merged_content: 合併後的內容
            current_chunks: 當前包含的 chunks
            current_token_count: 當前 token 數
            merged_parts: 要更新的合併部分字典
            saved_parts: 要更新的已保存部分列表
            update_metadata: 是否更新 metadata
        """
        chroma_ids, chunk_ids = self._extract_chunk_ids(current_chunks)
        
        # 保存這個部分的資訊（深拷貝以避免後續修改影響）
        saved_parts.append({
            'file_key': file_key,
            'chunks': current_chunks.copy(),
            'content': merged_content,
            'token_count': current_token_count,
            'chunk_ids': chunk_ids.copy()
        })
        
        # 更新 metadata
        if update_metadata:
            self._update_document_group_metadata(chroma_ids, file_key)
        
        # 保存到 merged_parts 字典
        merged_parts[file_key] = {
            "content": merged_content,
            "total_token_count": current_token_count,
            "chunk_ids": chunk_ids
        }
    
    def _try_balance_last_part(self,
                               current_chunks: List[Dict[str, Any]],
                               merged_content: str,
                               last_part_token: int,
                               saved_parts: List[Dict[str, Any]],
                               max_token_per_group: int) -> Tuple[List[Dict[str, Any]], str, int]:
        """
        嘗試從前一部分借用 chunks 來平衡最後一部分的 token 數
        
        參數:
            current_chunks: 當前最後一部分的 chunks
            merged_content: 當前最後一部分的合併內容
            last_part_token: 當前最後一部分的 token 數
            saved_parts: 已保存的部分列表
            max_token_per_group: 每個分組的最大 token 數
            
        返回:
            tuple: (更新後的 current_chunks, 更新後的 merged_content, 更新後的 last_part_token)
        """
        if not (last_part_token < max_token_per_group and saved_parts):
            return current_chunks, merged_content, last_part_token
        
        logger.info(f'最後一部分 token 數較少，嘗試從前一部分由後往前補充 chunks')
        
        last_saved_part = saved_parts[-1]
        last_saved_chunks = last_saved_part['chunks']
        original_chunk_len = len(current_chunks)
        
        # 從後往前遍歷前一部分的 chunks，直接嘗試前置加入
        for i in range(len(last_saved_chunks) - 1, -1, -1):
            chunk = last_saved_chunks[i]
            incremental_token_count = chunk.get('token_count_without_overlap')
            new_segment = chunk.get('new_segment_without_overlap', '')
            
            if incremental_token_count is None:
                _, incremental_token_count = self._get_chunk_incremental_info(None, chunk)
            
            predicted_token_count = last_part_token + (incremental_token_count or 0)
            logger.info(f'測試從前一部分借用第 {i} 個 chunk，預測 token 數: {predicted_token_count}')
            
            if predicted_token_count <= max_token_per_group:
                # 直接把 chunk 前置到 current_chunks，並重新合併計算
                current_chunks.insert(0, chunk)
                merged_content = f"{new_segment}{merged_content}"
                last_part_token = predicted_token_count
                logger.info(f'借用此 chunk 後，預估 token: {last_part_token}')
            else:
                logger.info(f'借用此 chunk 會超過限制 ({predicted_token_count} > {max_token_per_group})，停止借用')
                break
        
        borrowed_count = len(current_chunks) - original_chunk_len
        if borrowed_count > 0:
            if last_part_token > max_token_per_group:
                logger.error(f'錯誤：最後一部分補充後超過限制！token 數: {last_part_token} > {max_token_per_group}')
            else:
                logger.info(f'✓ 最後一部分從前一部分借用 {borrowed_count} 個 chunks')
                logger.info(f'✓ 更新後：最後一部分 token 數 = {last_part_token} / {max_token_per_group} ({last_part_token/max_token_per_group*100:.1f}%)')
        else:
            logger.info(f'無法從前一部分借用任何 chunks（會導致超過限制）')
        
        return current_chunks, merged_content, last_part_token
    
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
                logger.warning(f"未找到 ID 的 metadata: {chroma_ids}")
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

            logger.info(f"已更新 {len(chroma_ids)} 個文件的 group_id: {group_id}")

        except Exception as e:
            logger.error(f"更新 ChromaDB 中的 group_id 時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    def process_all_documents(self, merge_remaining_by_similarity: bool = True, save_folder_path: str = None) -> Dict[str, Dict[str, Any]]:
        """
        檢索和處理 ChromaDB 內容的主要方法
        現在支持提取剩餘的零碎 chunks 並按相似度合併，並可選擇性保存結果
        
        參數:
            merge_remaining_by_similarity: 是否對剩餘的零碎 chunks 按相似度合併
            save_folder_path: 保存路徑（可選），如果為 None 則不保存
        
        返回:
            包含按來源文件組織的完整文件內容與 token 計數的字典
        """
        file_content_dict = self.get_chunk_content_from_chroma()
        file_whole_content = self.build_merged_files_with_chunk_splitting(file_content_dict, update_metadata=True)

        # 如果提供了保存路徑，保存處理結果
        if save_folder_path is not None:
            logger.info(f"保存處理結果到: {save_folder_path}")
            self._save_expanded_result(file_whole_content, save_folder_path)
        
        return file_whole_content

    def _save_expanded_result(self, expanded_content: Dict[str, Dict[str, Any]], save_folder_path: str) -> None:
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
                    logger.info(f"已保存合併文件: {merged_filepath}")
                    total_files_saved += 1
                except Exception as e:
                    logger.error(f"保存文件失敗 {merged_filepath}: {e}")
        
        # 保存統計資料為 JSON
        json_filepath = os.path.join(save_folder_path, "statistics.json")
        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, ensure_ascii=False, indent=2)
            self._ensure_file_permissions(json_filepath)
            logger.info(f"已保存統計資料: {json_filepath}")
        except Exception as e:
            logger.error(f"保存統計資料失敗: {e}")
        
        logger.info(f"總共處理了 {len(expanded_content)} 個文件，保存了 {total_files_saved} 個合併文件")

    def _combine_small_files_by_similarity_into_merged_parts(self, 
                                                             small_files: List[Tuple[str, List[Dict[str, Any]], int]], 
                                                             update_metadata: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        按相似度合併小檔案
        
        參數:
            small_files: 小檔案列表，每個元素為 (source_file, content_list, file_total_token)
            update_metadata: 是否更新 ChromaDB metadata
            
        返回:
            合併後的文件內容字典
        """
        merged_parts = {}

        # If no small files, return empty dict
        if not small_files:
            return merged_parts

        logger.info(f'combine small files: {small_files[0][0]}')

        # If only one file, process it directly and return
        if len(small_files) == 1:
            source_file, content_list, file_total_token = small_files[0]
            merged_content = self._get_file_content_without_overlap(content_list)
            current_chunks = content_list
            
            # 移除副檔名以避免 .txt.txt 問題
            file_key = self._normalize_filename(source_file)
            
            # Collect chroma_ids and chunk_ids
            chroma_ids, chunk_ids = self._extract_chunk_ids(current_chunks)
            
            # update metadata (使用 file_key 而非 source_file)
            if update_metadata:
                self._update_document_group_metadata(chroma_ids, file_key)
            
            # Save merged_content to merged_parts dictionary
            merged_parts[file_key] = {
                "content": merged_content,
                "total_token_count": file_total_token,
                "chunk_ids": chunk_ids
            }
            
            return merged_parts

        # calculate similarity matrix for small files
        n = len(small_files)
        score_matrix = np.zeros((n, n))

        for i in range(n):
            score_matrix[i, i] = 1.0
            for j in range(i+1, n):
                score = self._calc_similarity_by_chunks_for_two_files(
                            small_files[i], 
                            small_files[j]
                        )
                score_matrix[i, j] = score
                score_matrix[j, i] = score

        # Multiple files need to be merged
        part_counter = 1
        _small_files = small_files[:]
        
        remaining_file_idx = list(range(len(_small_files)))
        while remaining_file_idx:
            merged_content = ''
            current_chunks = []
            accumulate_token = 0

            # Start from the first file
            base_idx = remaining_file_idx[0]
            base_file = _small_files[base_idx]
            source_file = base_file[0]
            content_list = base_file[1]
            file_total_token = base_file[2]

            # 使用不包含 overlap 的內容生成 merged file
            merged_content = self._get_file_content_without_overlap(content_list)
            accumulate_token = file_total_token
            current_chunks.extend(content_list)

            to_remove_indices = [base_idx]

            similarity_scores = []
            for other_idx in remaining_file_idx:
                if other_idx != base_idx:
                    score = score_matrix[base_idx, other_idx]
                    similarity_scores.append((other_idx, score))

            # sort by similarity
            similarity_scores.sort(key=lambda x: x[1], reverse=True)

            # merge files by similarity
            for other_idx, score in similarity_scores:
                other_file = _small_files[other_idx]
                other_file_token = other_file[2]

                # if adding this file would exceed the limit, stop merging
                if accumulate_token + other_file_token >= settings.MAX_TOKENS_PER_GROUP:
                    continue

                # merge content (使用不包含 overlap 的內容)
                other_file_content = self._get_file_content_without_overlap(other_file[1])
                merged_content += '\n\n' + other_file_content
                accumulate_token += other_file_token
                current_chunks.extend(other_file[1])
                to_remove_indices.append(other_idx)

            # generate file_key
            if len(to_remove_indices) == 1:
                # 移除副檔名以避免 .txt.txt 問題
                file_key = self._normalize_filename(source_file)
            else:
                file_key = f'small_files_merged_{part_counter}'
                part_counter += 1

            # collect chroma_ids and chunk_ids
            chroma_ids = [chunk.get('chroma_id') for chunk in current_chunks if chunk.get('chroma_id')]
            chunk_ids = [chunk.get('chunk_id') for chunk in current_chunks if chunk.get('chunk_id')]

            # update metadata
            if update_metadata:
                self._update_document_group_metadata(chroma_ids, file_key)

            # Save merged_content to merged_parts dictionary
            merged_parts[file_key] = {
                'content': merged_content,
                'total_token_count': accumulate_token,
                # 'chroma_ids': chroma_ids,
                'chunk_ids': chunk_ids
            }

            for idx in to_remove_indices:
                remaining_file_idx.remove(idx)

        return merged_parts




    
    

