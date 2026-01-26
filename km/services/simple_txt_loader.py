"""
Simple TXT file loader
Used to process local txt files
"""
import os
from pathlib import Path
from typing import List
from loguru import logger
from langchain_community.docstore.document import Document



class SimpleTxtLoader:
    """Simple TXT file loader - handles local txt files only"""
    
    def __init__(self):
        self.supported_extensions = {'.txt'}
        logger.info("SimpleTxtLoader initialized")
    
    def load_txt_files(self, file_path_list: List[str]) -> List[Document]:
        """
        Load a list of local txt files
        
        Args:
            file_path_list: List of local file paths
            
        Returns:
            List[Document]
        """
        documents = []
        
        for file_path in file_path_list:
            try:
                doc = self._load_single_file(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load file: {file_path}, error: {str(e)}")
                continue
        
        logger.info(f"SimpleTxtLoader loaded {len(documents)} files")
        return documents

    def load_file_info_list(self, file_info_list) -> List[Document]:
        """
        Load a list of FileInfo objects
        
        Args:
            file_info_list: List of FileInfo objects with path and filename
            
        Returns:
            List[Document]
        """
        documents = []
        
        for file_info in file_info_list:
            try:
                doc = self._load_single_file_with_name(file_info.path, file_info.filename)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load file: {file_info.path} (filename: {file_info.filename}), error: {str(e)}")
                continue
        
        logger.info(f"SimpleTxtLoader loaded {len(documents)} files (FileInfo)")
        return documents
    
    def _load_single_file(self, file_path: str) -> Document:
        """
        Load a single local file
        
        Args:
            file_path: Local file path
            
        Returns:
            Document
        """
        
        # Check file existence
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return None
        
        # Check file extension
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {file_path} (extension: {file_extension})")
            return None
        
        # Use full filename instead of file_stem
        filename = Path(file_path).name
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"File content is empty: {file_path}")
                return None
            
            # Create Document object
            document = Document(
                page_content=content,
                metadata={
                    'source': filename,
                    'source_file': filename,  # 添加 source_file 以保持与外部解析器的一致性
                    'file_path': file_path,
                    'loader': 'SimpleTxtLoader',
                }
            )
            
            logger.info(f"Loaded file: {file_path} ({len(content)} chars)")
            return document
            
        except UnicodeDecodeError:
            # Try another encoding
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read().strip()
                
                if content:
                    document = Document(
                        page_content=content,
                        metadata={
                            'source': filename,
                            'source_file': filename,
                            'file_path': file_path,
                            'file_type': file_extension,
                            'loader': 'simple_txt_loader',
                            'processing_mode': 'easy',
                            'source_type': 'local',
                            'encoding': 'gbk'
                        }
                    )
                    
                    logger.info(f"Loaded file (GBK encoding): {file_path} ({len(content)} chars)")
                    return document
                    
            except Exception as e:
                logger.error(f"Reading file failed (multiple encodings tried): {file_path}, error: {str(e)}")
                return None
        
        except Exception as e:
            logger.error(f"Reading file failed: {file_path}, error: {str(e)}")
            return None

    def _load_single_file_with_name(self, file_path: str, filename: str) -> Document:
        """
        Load a single local file (with specified filename)
        
        Args:
            file_path: Local file path
            filename: Custom filename (with extension)
            
        Returns:
            Document
        """
        
        # Check file existence
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return None
        
        # Check file extension (based on actual file path)
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {file_path} (extension: {file_extension})")
            return None
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"File content is empty: {file_path}")
                return None
            
            # Create Document object with specified filename
            document = Document(
                page_content=content,
                metadata={
                    'source': filename,
                    'source_file': filename,  # 添加 source_file 以保持与外部解析器的一致性
                    'file_path': file_path,
                    'loader': 'simple_txt_loader',
                    'file_type': file_extension,
                }
            )
            
            logger.info(f"Loaded file: {file_path} (as: {filename}) ({len(content)} chars)")
            return document
            
        except UnicodeDecodeError:
            # Try another encoding
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read().strip()
                
                if content:
                    document = Document(
                        page_content=content,
                        metadata={
                            'source': filename,
                            'source_file': filename,
                            'file_path': file_path,
                            'file_type': file_extension,
                            'loader': 'simple_txt_loader',
                            'processing_mode': 'easy',
                            'source_type': 'local',
                            'encoding': 'gbk'
                        }
                    )
                    
                    logger.info(f"Loaded file (GBK encoding): {file_path} (as: {filename}) ({len(content)} chars)")
                    return document
                    
            except Exception as e:
                logger.error(f"Reading file failed (multiple encodings tried): {file_path} (as: {filename}), error: {str(e)}")
                return None
        
        except Exception as e:
            logger.error(f"Reading file failed: {file_path} (as: {filename}), error: {str(e)}")
            return None
    
    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if the file is supported
        
        Args:
            file_path: File path
            
        Returns:
            bool
        """
        file_extension = Path(file_path).suffix.lower()
        return file_extension in self.supported_extensions
    
    def filter_supported_files(self, file_path_list: List[str]) -> List[str]:
        """
        Filter supported local files
        
        Args:
            file_path_list: List of local file paths
            
        Returns:
            List of supported file paths
        """
        supported_files = []
        unsupported_files = []
        
        for file_path in file_path_list:
            if self.is_supported_file(file_path):
                supported_files.append(file_path)
            else:
                unsupported_files.append(file_path)
        
        if unsupported_files:
            logger.warning(f"Found {len(unsupported_files)} unsupported files, will skip: {unsupported_files}")
        
        logger.info(f"Filtered {len(supported_files)} supported files")
        return supported_files
    
    def cleanup(self):
        """Cleanup method (kept for interface compatibility)"""
        pass
    
    def __del__(self):
        """Destructor"""
        pass 