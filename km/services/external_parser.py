"""
External parser service
Calls external tools/APIs to parse files and returns raw parsed results
"""

import os
import uuid
import requests
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
from langchain_community.docstore.document import Document

from config import settings


class ExternalParserService:
    """External parser service - focused on API calls"""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or getattr(settings, 'DOCUMENT_ANALYSIS_URL', 'http://localhost:5000/api/v1/analyze')
        self.api_key = api_key or getattr(settings, 'DOCUMENT_ANALYSIS_API_KEY', '')
        logger.info(f"ExternalParserService initialized: API={self.api_url}")
    
    async def parse_files(self, file_list) -> List[Document]:
        """
        Batch-parse a list of files
        
        Args:
            file_list: FileInfo objects or path strings
            
        Returns:
            List[Document]
        """
        logger.info(f"Start batch parsing {len(file_list)} files")
        
        all_documents = []
        successful_count = 0
        failed_count = 0
        
        for file_item in file_list:
            try:
                # Support FileInfo objects and string paths
                if hasattr(file_item, 'path') and hasattr(file_item, 'filename'):
                    # FileInfo object
                    file_path = file_item.path
                    filename = file_item.filename
                    documents = await self.parse_single_file_with_name(file_path, filename)
                else:
                    # String path (backwards compatible)
                    file_path = file_item
                    documents = await self.parse_single_file(file_path)
                    
                all_documents.extend(documents)
                successful_count += 1
                logger.info(f"Parsed file successfully: {file_path}, returned raw parsed data")
                
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to parse file: {file_path if hasattr(file_item, 'path') else file_item}, error: {str(e)}")
                continue
        
        logger.info(f"Batch parsing completed: success {successful_count}, failed {failed_count}, total {len(all_documents)} documents")
        return all_documents
    
    async def parse_single_file(self, file_path: str) -> List[Document]:
        """
        Parse a single file
        
        Args:
            file_path: File path
            
        Returns:
            List[Document] containing raw parsed data
        """
        logger.info(f"Start parsing file: {file_path}")
        
        # Validate file existence
        # if not os.path.exists(file_path):
        #     raise FileNotFoundError(f"File does not exist: {file_path}")
        
        file_path_obj = Path(file_path)
        file_name = file_path_obj.name
        file_stem = file_path_obj.stem  # filename without extension
        flow_id = str(uuid.uuid4())
        
        try:
            # Call external parsing API
            response_data = await self._call_external_api(file_path, flow_id)
            
            # Convert raw parsed data to Documents, preserve page structure
            documents = self._convert_raw_data_to_documents(response_data, file_name, file_path, flow_id)
            
            logger.info(f"File parsed: {file_path}, generated {len(documents)} page documents")
            return documents
            
        except Exception as e:
            logger.error(f"File parsing failed: {file_path}, error: {str(e)}")
            raise

    async def parse_single_file_with_name(self, file_path: str, filename: str) -> List[Document]:
        """
        Parse a single file (custom filename)
        
        Args:
            file_path: File path
            filename: Custom filename (with extension)
            
        Returns:
            List[Document] with raw parsed data
        """
        logger.info(f"Start parsing file: {file_path}, using filename: {filename}")
        
        flow_id = str(uuid.uuid4())
        
        try:
            # Call external parsing API
            response_data = await self._call_external_api(file_path, flow_id)
            
            # Convert raw parsed data to Documents, using specified filename
            documents = self._convert_raw_data_to_documents(response_data, filename, file_path, flow_id)
            
            logger.info(f"File parsed: {file_path} (filename: {filename}), generated {len(documents)} page documents")
            return documents
            
        except Exception as e:
            logger.error(f"File parsing failed: {file_path}, error: {str(e)}")
            raise
    
    def _convert_raw_data_to_documents(
        self, 
        content_data: Dict[str, Any], 
        filename: str, 
        file_path: str,
        flow_id: str
    ) -> List[Document]:
        """
        Convert raw parsed data to Documents while preserving page structure
        
        Args:
            content_data: Raw content data from API
            filename: Filename (with extension)
            file_path: Original file path
            flow_id: Flow ID
            
        Returns:
            List[Document], each representing a page or page range
        """
        documents = []
        
        try:
            if not isinstance(content_data, dict):
                logger.warning(f"Content data is not a dict: {type(content_data)}")
                # Not a dict: create a single document
                page_content = str(content_data) if content_data else ""
                doc = Document(
                    page_content=page_content,
                    metadata={
                        'source': filename,
                        'source_file': filename,
                        'original_file_path': file_path,
                        'parsed_by': 'external_api',
                        'api_url': self.api_url,
                        'flow_id': flow_id,
                        'page_key': 'unknown',
                        'parent_content': page_content
                    }
                )
                documents.append(doc)
                return documents
            
            content_cache = []
            # Dict format: each key-value pair represents a page or page range
            for page_key, page_content in content_data.items():
                if not isinstance(page_content, str) or not page_content.strip():
                    continue
                
                content_cache.append(page_content.strip())
                
            page_content = "\n\n\n".join(content_cache)
            doc = Document(
                page_content=page_content,
                metadata={
                    'source': filename,
                    'source_file': filename,
                    'file_path': file_path,
                    'loader': 'ExternalParserService',
                    'api_url': self.api_url,
                    'flow_id': flow_id,
                }
            )
            documents.append(doc)
            
            if not documents:
                logger.warning(f"No valid content extracted from file {filename}")
                # No content extracted: create an empty document
                doc = Document(
                    page_content="",
                    metadata={
                        'source': filename,
                        'source_file': filename,
                        'file_path': file_path,
                        'parsed_by': 'external_api',
                        'api_url': self.api_url,
                        'flow_id': flow_id,
                    }
                )
                documents.append(doc)
            
            logger.info(f"Extracted {len(documents)} pages from file {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"Converting data failed: {filename}, error: {str(e)}")
            # Create an error document
            doc = Document(
                page_content="",
                metadata={
                    'source': filename,
                    'source_file': filename,
                    'file_path': file_path,
                    'loader': 'external_api',
                    'api_url': self.api_url,
                    'flow_id': flow_id,
                }
            )
            return [doc]

    def _extract_uuid_from_url(self, file_path: str) -> str:
        """
        Extract UUID from URL and append it as &id parameter
        
        Args:
            file_path: URL in format http://{IP}:{port}/{UUID}?test=1&filePath=filename
            
        Returns:
            URL with &id={UUID} appended
        """
        try:
            # Pattern to match UUID in URL: http://IP:port/UUID?...
            # UUID can contain commas and other characters based on your example
            pattern = r'http://[^/]+/([^?]+)\?'
            match = re.search(pattern, file_path)
            
            if match:
                uuid_part = match.group(1)
                # Append &id={UUID} to the URL
                return f"{file_path}&id={uuid_part}"
            else:
                # If no UUID found in URL, keep original behavior
                logger.warning(f"Could not extract UUID from URL: {file_path}")
                return f"{file_path}&id=123"
                
        except Exception as e:
            logger.error(f"Error extracting UUID from URL {file_path}: {str(e)}")
            return f"{file_path}&id=123"
    
    async def _call_external_api(self, file_path: str, flow_id: str = "test") -> Dict[str, Any]:
        """
        Call external parsing API
        
        Args:
            file_path: File path (MinIO path)
            flow_id: Flow ID
            
        Returns:
            API response data
        """
        # Build request payload
        # Extract UUID from URL and replace &id=123 with the actual UUID
        file_path_with_id = self._extract_uuid_from_url(file_path)
        request_data = {
            "flow_id": flow_id,
            "remote_storage_url": file_path_with_id,
            "Minio_Path": file_path_with_id,
            "guid": flow_id,
            "Prompt": "",
            "is_rag": True
        }
        
        headers = {
            "accept": "application/json",
            "Authorization": self.api_key
        }
        
        try:
            logger.info(f"Calling external parsing API: {self.api_url}, file: {file_path}")
            
            response = requests.post(
                self.api_url, 
                json=request_data, 
                headers=headers,
                timeout=30000  # 5-minute timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed: HTTP {response.status_code}, {response.text}")
            
            response_data = response.json()
            # logger.info("\n"*10)
            # logger.info(response_data)
            # logger.info("\n"*10)
            # Check response format
            if "code" not in response_data or response_data["code"] != 200:
                error_msg = response_data.get('error', response_data.get('message', 'unknown error'))
                raise Exception(f"API returned error: {error_msg}")
            
            if "data" not in response_data or "content" not in response_data["data"]:
                raise Exception("API response format error: missing data.content")
            
            logger.info(f"External API call succeeded: {file_path}")
            return response_data["data"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network request error: {str(e)}")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    def validate_api_connection(self) -> bool:
        """
        Validate external API connectivity
        
        Returns:
            Whether the connection is healthy
        """
        try:
            # Send a simple health check
            response = requests.get(self.api_url.replace('/analyze', '/health'), timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get API information
        
        Returns:
            API info dict
        """
        return {
            'api_url': self.api_url,
            'is_connected': self.validate_api_connection(),
            'supported_formats': ['pdf', 'docx', 'xlsx', 'txt', 'csv']
        } 