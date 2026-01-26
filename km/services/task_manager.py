"""
Task management service - lightweight (no database)
Coordinates the document processing workflow
"""
import os
from typing import List
from pathlib import Path
from loguru import logger

try:
    # Try relative imports (when imported as a module)
    from .document_processor import DocumentProcessor, ProcessingConfig
    from .kvcache_generator import KVCacheGeneratorService
    from .external_parser import ExternalParserService
    from .simple_txt_loader import SimpleTxtLoader
except ImportError:
    # Use absolute imports when running directly
    from document_processor import DocumentProcessor, ProcessingConfig
    from kvcache_generator import KVCacheGeneratorService
    from external_parser import ExternalParserService
    from simple_txt_loader import SimpleTxtLoader
from langchain_community.docstore.document import Document


class TaskManagerService:
    """Task management service - lightweight"""
    
    def __init__(self, base_folder: str = "./data"):
        self.base_folder = base_folder
        self.document_processor = None  # lazy initialization
        self.kvcache_generator = None   # lazy initialization
        self.external_parser = ExternalParserService()
        self.simple_txt_loader = SimpleTxtLoader()  # simple mode text loader
        logger.info("TaskManagerService initialized (no-database mode)")
    
    def _prepare_collection_folder(self, collection_folder: str):
        """
        Prepare the collection folder: clear existing contents and create required subdirectories
        
        Args:
            collection_folder: Path to the collection folder
        """
        import shutil
        
        logger.info(f"Preparing collection folder: {collection_folder}")
        
        # If the folder exists, clear it first
        if os.path.exists(collection_folder):
            logger.info(f"Removing existing collection folder: {collection_folder}")
            shutil.rmtree(collection_folder)
        
        # Create the main folder and subdirectories
        subdirectories = [
            "file_content",     # original file contents
            "processed_output"  # processed output documents
        ]
        
        for subdir in subdirectories:
            subdir_path = os.path.join(collection_folder, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            logger.debug(f"Created subdirectory: {subdir_path}")
        
        logger.info(f"Collection folder structure initialized: {collection_folder}")
        logger.info("Directory structure:")
        logger.info(f"  {collection_folder}/")
        for subdir in subdirectories:
            logger.info(f"  ├── {subdir}/")

    def initialize_for_collection(self, collection_name: str, config: ProcessingConfig = None):
        """Initialize processors for a specific collection"""
        # Clean collection name: remove leading/trailing spaces and ensure valid characters
        collection_name = collection_name.strip()
        logger.info(f"Initialize processors for collection '{collection_name}'")
        
        # Create config
        if config is None:
            config = ProcessingConfig()
        config.collection_name = collection_name
        
        # Initialize processors
        self.document_processor = DocumentProcessor(config, self.base_folder)
        self.kvcache_generator = KVCacheGeneratorService(collection_name, self.base_folder)
        
        logger.info(f"Processors initialized for collection '{collection_name}'")
    
    async def _load_documents_from_paths(self, file_list) -> List[Document]:
        """
        Load documents from file list
        Automatically select processing mode based on file source:
        - MinIO/remote paths: use external parser (supports various formats)
        - Local paths: use simple mode (txt files only)
        """
        
        # Classify files
        local_files = []
        remote_files = []
        
        for file_info in file_list:
            file_path = file_info.path
            if (file_path.startswith('s3://') or 
                file_path.startswith('minio://') or 
                file_path.startswith('http://') or 
                file_path.startswith('https://')):
                remote_files.append(file_info)
            else:
                local_files.append(file_info)
        
        all_documents = []
        
        # Process local files (simple mode, txt only)
        if local_files:
            logger.info(f"Processing {len(local_files)} local txt files in simple mode")
            
            # Use the new load_file_info_list method for FileInfo objects
            local_documents = self.simple_txt_loader.load_file_info_list(local_files)
            all_documents.extend(local_documents)
            logger.info(f"Local file processing completed, loaded {len(local_documents)} documents")
        
        # Process remote files (external parser, various formats supported)
        if remote_files:
            logger.info(f"Processing {len(remote_files)} remote files with external parser")
            
            # Use external parser to process files in batch (pass FileInfo objects)
            remote_documents = await self.external_parser.parse_files(remote_files)
            all_documents.extend(remote_documents)
            logger.info(f"Remote file processing completed, generated {len(remote_documents)} documents")
        
        logger.info(f"Total loaded documents: {len(all_documents)}")
        return all_documents
    
    async def process_collection_workflow(self, collection_name: str, input_files, language: str = "zh-TW") -> dict:
        """
        Full collection processing workflow
        First use document_processor to get merged content, then use kvcache_generator to generate KV cache
        Start PhisonAI only once per collection, then process all merged files in a loop
        
        Automatically select processing mode based on file source:
        - MinIO/remote paths: external parser (supports various formats)
        - Local paths: simple mode (txt files only)
        
        Args:
            collection_name: Collection name
            input_files: List of input files (FileInfo objects)
            language: Language for prompt template (zh-TW, en, english)
            
        Returns:
            Result dict
        """
        try:
            # Clean collection name: remove leading/trailing spaces
            collection_name = collection_name.strip()
            logger.info(f"Start processing collection: {collection_name}")
            
            # Step 0: Create and clear collection folder
            collection_folder = os.path.join(self.base_folder, collection_name)
            # self._prepare_collection_folder(collection_folder)
            
            # Step 1: Initialize collection processors
            config = ProcessingConfig(collection_name=collection_name)
            self.initialize_for_collection(collection_name, config)
            
            # Step 2: Process documents by document_processor and get merged files
            logger.info("Step 1: Document processing and merging...")
            
            # Load documents
            documents = await self._load_documents_from_paths(input_files)
            if not documents:
                raise ValueError("No documents were successfully loaded")
            
            # Process documents (this will generate merged files)
            # Use collection folder as file_content_folder
            task_id = f"{collection_name}_task"
            result, chunked_documents= self.document_processor.process_documents(documents, task_id, collection_name)
            
            # Get merged files
            merged_files = self.document_processor.list_merged_files()
            merged_filenames = self.document_processor.list_merged_filenames()
            logger.info(f"Document processing completed, got {len(merged_files)} merged files")
            
            if not merged_files:
                logger.warning("No merged files were generated")
                return {
                    "collection_name": collection_name,
                    "input_files_count": len(input_files),
                    "documents_count": len(documents),
                    "merged_files": [],
                    "merged_filenames": [],
                    "kvcache_paths": []
                }
            
            # Step 3: Use kvcache_generator to generate KV cache for all merged files
            logger.info("Step 2: Generating KV cache...")
            
            # Generate KV cache for entire collection (start PhisonAI once, process all files in a loop)
            processed_count = await self.kvcache_generator.generate_kvcaches_for_collection(
                collection_name=collection_name,
                merged_files=merged_files,
                language=language
            )
            
            # logger.info(f"KV cache generation completed, processed {processed_count} files successfully")
            
            # Step 4: Create and save BM25 index for this collection
            # 注意：必須在 process_documents 完成後才創建 BM25 索引
            # 因為需要從 ChromaDB 讀取已包含 group_id 的文檔
            logger.info("Step 3: Creating BM25 index...")
            try:
                from services.bm25_index_manager import BM25IndexManager
                bm25_manager = BM25IndexManager()
                
                # 檢查 BM25 是否可用
                if not bm25_manager.is_available():
                    logger.warning("BM25 dependencies not available. Skipping BM25 index creation.")
                else:
                    # 根據 language 參數決定使用的分詞器
                    # 將 language 從 config 格式轉換為 BM25 格式
                    bm25_language = 'zh-TW'  # 默認中文
                    if language in ['en-US', 'english', 'en']:
                        bm25_language = 'en-US'
                    elif language in ['ja-JP', 'japanese', 'jp', 'ja']:
                        bm25_language = 'ja-JP'
                    
                    # 從 ChromaDB 讀取文檔來建立 BM25 索引
                    # 此時文檔已經包含 group_id（由 process_documents 中的 process_all_documents 添加）
                    # 獲取 vectorstore（從 document_processor）
                    vectorstore = self.document_processor.kv_cache_handler.chroma if self.document_processor.kv_cache_handler else None
                    
                    if vectorstore:
                        if bm25_manager.create_index_from_chroma(vectorstore, collection_name, bm25_language):
                            bm25_index_path = os.path.join(self.base_folder, collection_name, "BM25_indices")
                            logger.info(f"BM25 index created and saved for collection {collection_name} at {bm25_index_path} (language: {bm25_language})")
                        else:
                            logger.warning(f"Failed to create BM25 index for collection {collection_name}")
                    else:
                        logger.warning("Vectorstore not available for BM25 index creation")
            except Exception as e:
                logger.error(f"Error creating BM25 index: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                # 不影響整體流程，繼續執行
                
            # Show results
            logger.info("Processing result:")
            logger.info(f"  - Collection: {collection_name}")
            logger.info(f"  - Input file count: {len(input_files)}")
            logger.info(f"  - Documents count: {len(documents)}")
            logger.info(f"  - Merged files count: {len(merged_files)}")
            logger.info(f"  - Merged filenames: {merged_filenames}")
            logger.info(f"  - KV Cache processed count: {processed_count}")
            
            return {
                "collection_name": collection_name,
                "input_files_count": len(input_files),
                "documents_count": len(documents),
                "merged_files": merged_files,
                "merged_filenames": merged_filenames,
                "kvcache_processed_count": processed_count,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Processing collection {collection_name} failed: {str(e)}")
            return {
                "collection_name": collection_name,
                "input_files_count": len(input_files) if input_files else 0,
                "error": str(e),
                "success": False
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.kvcache_generator:
            await self.kvcache_generator.close()
        if self.simple_txt_loader:
            self.simple_txt_loader.cleanup()
        logger.info("TaskManagerService cleanup completed")


# Usage example
async def main_example():
    """Usage example"""
    import asyncio
    
    # Create task manager (using configured base path)
    try:
        from ..config import settings
        base_folder = settings.BASE_FOLDER
    except ImportError:
        from config import settings
        base_folder = settings.BASE_FOLDER
    
    task_manager = TaskManagerService(base_folder=base_folder)
    
    try:
        # Process a collection
        result = await task_manager.process_collection_workflow(
            collection_name="test_collection",
            input_files=[
                "/home/vscode/sh/agent_builder/test/data/test1.txt",
                "/home/vscode/sh/agent_builder/test/data/test2.txt", 
                "/home/vscode/sh/agent_builder/test/data/test3.txt"
            ]
        )
        
        if result["success"]:
            logger.info("Workflow completed!")
        else:
            logger.error(f"Workflow failed: {result.get('error')}")
            
    finally:
        await task_manager.cleanup()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_example()) 