"""
BM25 Index Manager
獨立管理 BM25 索引的建立、載入、儲存和搜尋功能
"""
import os
import pickle
import heapq
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
import sys

# 添加父目錄到路徑以便導入 config
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from config import settings

from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document

# BM25 imports
try:
    from rank_bm25 import BM25Okapi
    import jieba
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("BM25 dependencies not available. Install with: pip install rank-bm25 jieba")

# Japanese tokenizer
try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False
    logger.warning("Janome not available for Japanese tokenization. Install with: pip install janome")


class BM25IndexManager:
    """BM25 索引管理器 - 負責索引的建立、載入、儲存和搜尋"""
    
    def __init__(self):
        """初始化 BM25 索引管理器"""
        self.bm25_cache = {}  # 儲存各 collection 的 BM25 索引和文檔
        self.janome_tokenizer = JanomeTokenizer() if JANOME_AVAILABLE else None
    
    def is_available(self) -> bool:
        """檢查 BM25 是否可用"""
        return BM25_AVAILABLE
    
    def tokenize_text(self, text: str, language: str = 'zh') -> List[str]:
        """
        文本分詞 - 支援中文、英文、日文
        
        Args:
            text: 要分詞的文本
            language: 語言代碼 ('zh' 中文, 'ja' 日文, 'en' 英文)
            
        Returns:
            分詞後的 token 列表
        """
        if not text:
            return []
        
        # 日文分詞
        if language == 'ja-JP' or language == 'jp':
            if not JANOME_AVAILABLE:
                logger.warning("Janome not available, falling back to simple split")
                return text.split()
            
            # 使用 janome 進行日文分詞
            tokens = [token.surface for token in self.janome_tokenizer.tokenize(text)]
            # 過濾掉空白和單字符標點符號
            tokens = [token for token in (t.strip() for t in tokens) if token and len(token) > 0]
            return tokens
        
        # 英文分詞（簡單空格分割）
        elif language == 'en-US':
            tokens = text.split()
            # 過濾掉空白
            tokens = [token.strip() for token in tokens if token.strip()]
            return tokens
        
        # 中文分詞（默認）
        else:
            if not BM25_AVAILABLE:
                return text.split()
            
            # 使用 jieba 進行中文分詞
            tokens = jieba.lcut(text)
            # 過濾掉空白和標點符號（優化：只調用 strip() 一次）
            tokens = [token for token in (t.strip() for t in tokens) if token and len(token) > 1]
            return tokens
    
    def load_index(self, chroma: Chroma, collection_name: str) -> bool:
        """
        載入 BM25 索引 - 優先從快取載入，如果不存在則從 ChromaDB 建立
        
        Args:
            chroma: Chroma 向量資料庫實例
            collection_name: Collection 名稱
            
        Returns:
            是否成功載入索引
        """
        if not BM25_AVAILABLE:
            logger.error("BM25 not available")
            raise RuntimeError("BM25 not available")
        
        # # 檢查是否已經在快取中
        # if collection_name in self.bm25_cache:
        #     logger.debug(f"BM25 index for collection {collection_name} already loaded from cache")
        #     return True
        
        try:
            # 嘗試從檔案載入已儲存的索引
            if self._load_index_from_file(collection_name):
                logger.info(f"BM25 index loaded from file for collection {collection_name}")
                return True
            
            # 如果檔案不存在，從 ChromaDB 建立新索引
            logger.info(f"Creating new BM25 index for collection {collection_name}")
            return self.create_index_from_chroma(chroma, collection_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index for collection {collection_name}: {str(e)}")
            raise
    
    def _load_index_from_file(self, collection_name: str) -> bool:
        """
        從檔案載入 BM25 索引
        
        Args:
            collection_name: Collection 名稱
            
        Returns:
            是否成功載入
        """
        try:
            # 構建索引檔案路徑（使用 BM25_indices 子目錄）
            bm25_dir = os.path.join(settings.BASE_FOLDER, collection_name, "BM25_indices")
            index_file = os.path.join(bm25_dir, f"{collection_name}_bm25.pkl")
            
            if not os.path.exists(index_file):
                logger.debug(f"BM25 index file not found: {index_file}")
                return False
            
            # 載入索引
            with open(index_file, 'rb') as f:
                bm25_data = pickle.load(f)
            
            # 儲存到快取
            self.bm25_cache[collection_name] = {
                'index': bm25_data['index'],
                'documents': bm25_data['documents'],
                'language': bm25_data.get('language', 'zh'),  # 向後兼容，默認為中文
                'created_at': bm25_data.get('created_at', 'unknown')
            }
            
            logger.info(f"BM25 index loaded from file: {index_file}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load BM25 index from file: {str(e)}")
            raise
    
    def create_index_from_documents(self, documents: List[Document], collection_name: str, language: str = 'zh') -> bool:
        """
        從 Document 列表直接建立 BM25 索引
        
        Args:
            documents: LangChain Document 列表
            collection_name: Collection 名稱
            language: 語言代碼 ('zh' 中文, 'ja' 日文, 'en' 英文)
            
        Returns:
            是否成功建立索引
        """
        if not BM25_AVAILABLE:
            logger.error("BM25 not available")
            raise RuntimeError("BM25 not available")
        
        try:
            if not documents:
                logger.warning(f"No documents provided for collection {collection_name}")
                raise ValueError(f"No documents provided for collection {collection_name}")
            
            logger.info(f"Creating BM25 index for language: {language}")
            
            # 為每個文檔創建分詞後的文本
            tokenized_docs = []
            bm25_documents = []
            
            for doc in documents:
                # 結合文檔內容和元數據進行分詞
                full_text = doc.page_content
                if doc.metadata and 'source' in doc.metadata:
                    full_text += f" {doc.metadata['source']}"
                
                tokenized_doc = self.tokenize_text(full_text, language=language)
                tokenized_docs.append(tokenized_doc)
                bm25_documents.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata.copy() if doc.metadata else {},
                    'tokens': tokenized_doc
                })
            
            # 創建 BM25 索引
            bm25_index = BM25Okapi(tokenized_docs)
            
            # 儲存到快取（包含語言信息）
            self.bm25_cache[collection_name] = {
                'index': bm25_index,
                'documents': bm25_documents,
                'language': language,
                'created_at': datetime.now().isoformat()
            }
            
            # 儲存到檔案
            self._save_index_to_file(collection_name)
            
            logger.info(f"BM25 index created from documents for collection {collection_name} with {len(documents)} documents (language: {language})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create BM25 index from documents: {str(e)}")
            raise
    
    def create_index_from_chroma(self, chroma: Chroma, collection_name: str, language: str = 'zh') -> bool:
        """
        從 ChromaDB 建立 BM25 索引
        
        Args:
            chroma: Chroma 向量資料庫實例
            collection_name: Collection 名稱
            
        Returns:
            是否成功建立索引
        """
        try:
            # 獲取所有文檔
            all_docs = chroma.get()
            documents = all_docs['documents']
            metadatas = all_docs['metadatas']
            
            if not documents:
                logger.warning(f"No documents found in collection {collection_name}")
                raise ValueError(f"No documents found in collection {collection_name}")
            
            # 為每個文檔創建分詞後的文本
            tokenized_docs = []
            bm25_documents = []
            
            for i, doc in enumerate(documents):
                # 結合文檔內容和元數據進行分詞
                full_text = doc
                # if metadatas and i < len(metadatas) and metadatas[i]:
                #     metadata = metadatas[i]
                #     if 'source' in metadata:
                #         full_text += f" {metadata['source']}"
                
                # 使用默認語言（中文）進行分詞
                tokenized_doc = self.tokenize_text(full_text, language=language)
                tokenized_docs.append(tokenized_doc)
                bm25_documents.append({
                    'content': doc,
                    'metadata': metadatas[i] if metadatas and i < len(metadatas) else {},
                    'tokens': tokenized_doc
                })
            
            # 創建 BM25 索引
            bm25_index = BM25Okapi(tokenized_docs)
            
            # 儲存到快取
            self.bm25_cache[collection_name] = {
                'index': bm25_index,
                'documents': bm25_documents,
                'language': language,  # ChromaDB 方法默認使用中文
                'created_at': datetime.now().isoformat()
            }
            
            # 儲存到檔案
            self._save_index_to_file(collection_name)
            
            logger.info(f"BM25 index created for collection {collection_name} with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create BM25 index from ChromaDB: {str(e)}")
            raise
    
    def _save_index_to_file(self, collection_name: str) -> bool:
        """
        將 BM25 索引儲存到檔案
        
        Args:
            collection_name: Collection 名稱
            
        Returns:
            是否成功儲存
        """
        try:
            if collection_name not in self.bm25_cache:
                logger.warning(f"No BM25 index in cache for collection {collection_name}")
                raise RuntimeError(f"BM25 index not found in cache for collection {collection_name}")
            
            # 構建目錄和檔案路徑（使用 BM25_indices 子目錄）
            bm25_dir = os.path.join(settings.BASE_FOLDER, collection_name, "BM25_indices")
            os.makedirs(bm25_dir, exist_ok=True)
            
            # 構建檔案路徑
            index_file = os.path.join(bm25_dir, f"{collection_name}_bm25.pkl")
            # 確認 index_file 是否存在，若存在則刪除
            if os.path.exists(index_file):
                os.remove(index_file)
            # 準備儲存資料
            bm25_data = self.bm25_cache[collection_name].copy()
            
            # 儲存到檔案
            with open(index_file, 'wb') as f:
                pickle.dump(bm25_data, f)
            
            logger.info(f"BM25 index saved to file: {index_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save BM25 index to file: {str(e)}")
            raise
    
    def search(self, chroma: Chroma, collection_name: str, question: str, k: int = 5, language: str = None) -> List[Dict]:
        """
        使用 BM25 進行搜尋
        
        Args:
            chroma: Chroma 向量資料庫實例（用於載入索引）
            collection_name: Collection 名稱
            question: 查詢問題
            k: 返回結果數量
            language: 查詢語言（如果為 None，使用索引創建時的語言）
            
        Returns:
            搜尋結果列表，每個結果包含 content, metadata, score
        """
        if not BM25_AVAILABLE:
            logger.error("BM25 not available")
            raise RuntimeError("BM25 not available")
        
        try:
            # 每次搜尋都先載入索引（發生錯誤時會丟出例外）
            self.load_index(chroma, collection_name)
            
            # 從快取獲取索引和文檔
            bm25_data = self.bm25_cache[collection_name]
            bm25_index = bm25_data['index']
            bm25_documents = bm25_data['documents']
            
            # 如果未指定語言，使用索引創建時的語言
            if language is None:
                language = bm25_data.get('language', 'zh')
            
            # 對查詢進行分詞
            query_tokens = self.tokenize_text(question, language=language)
            
            # 空查詢檢查
            if not query_tokens:
                logger.warning("Empty query tokens after tokenization")
                raise ValueError("Empty query after tokenization")
            
            # 使用 BM25 進行搜尋
            scores = bm25_index.get_scores(query_tokens)
            
            # 使用 heapq 高效獲取前 k 個結果（O(n log k) 而非 O(n log n)）
            # 注意：BM25 分數越高越好，所以使用負數來配合 heapq（默認是最小堆）
            top_items = heapq.nlargest(k, enumerate(scores), key=lambda x: x[1])
            
            results = []
            for idx, score in top_items:
                if score != 0:  # 只返回有分數的結果
                    doc_info = bm25_documents[idx]
                    results.append({
                        'content': doc_info['content'],
                        'metadata': doc_info['metadata'],
                        'score': score
                    })
            
            logger.info(f"BM25 search returned {len(results)} results (language: {language})")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            raise
    
    def clear_cache(self, collection_name: Optional[str] = None):
        """
        清除 BM25 緩存
        
        Args:
            collection_name: 要清除的 collection 名稱，如果為 None 則清除所有緩存
        """
        if collection_name:
            # 只清除特定 collection 的 BM25 緩存
            if collection_name in self.bm25_cache:
                del self.bm25_cache[collection_name]
                logger.info(f"Cleared BM25 cache for: {collection_name}")
        else:
            # 清除所有 BM25 緩存
            self.bm25_cache.clear()
            logger.info("Cleared all BM25 caches")

