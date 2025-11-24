# -*- coding: utf-8 -*-
"""
Simplified API - Document processing and KV Cache generation
"""
import os
import uuid
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from config import settings
from services.task_manager import TaskManagerService
from services.rag_query_service import RAGQueryService
from loguru import logger


# Data models
class FileInfo(BaseModel):
    """File information"""
    path: str = Field(..., description="File path")
    filename: str = Field(..., description="Filename (with extension)")

class ProcessingRequest(BaseModel):
    """Document processing request"""
    collection_name: str = Field(..., description="Collection name")
    file_list: List[FileInfo] = Field(..., description="File list, each file contains path and filename")
    language: str = Field(default="zh-TW", description="Language for prompt template (zh-TW, en, english)")
    

class ProcessingResponse(BaseModel):
    """Processing response"""
    task_id: str
    status: str
    collection_name: str
    input_files_count: int
    documents_count: Optional[int] = None
    merged_files_count: Optional[int] = None
    kvcache_processed_count: Optional[int] = None
    processing_time: Optional[float] = None
    
    # Input file stats
    local_files_count: Optional[int] = None
    minio_files_count: Optional[int] = None
    
    # Paths info
    collection_folder: Optional[str] = None
    file_content_dir: Optional[str] = None
    processed_output_dir: Optional[str] = None
    kvcache_dir: Optional[str] = None
    merged_files: Optional[List[str]] = None
    
    message: Optional[str] = None
    error: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str
    status: str
    message: Optional[str] = None


class TokenRequest(BaseModel):
    """Token calculate request"""
    file_path: str = Field(..., description="File path")
    filename: str = Field(..., description="File name (with extension)")


class TokenResponse(BaseModel):
    """Token calculate response"""
    status: bool
    taskID: str
    fileName: str
    token_count: int
    message: Optional[str] = None
    error: Optional[str] = None


# RAG Query related models
class QueryRequest(BaseModel):
    """RAG Query request"""
    collection_name: str = Field(..., description="Collection name to query")
    question: str = Field(..., description="User question")
    k: int = Field(default=5, description="Number of top-k results to retrieve")


class QueryResponse(BaseModel):
    """RAG Query response"""
    success: bool
    filename: Optional[str] = None
    file_path: Optional[str] = None
    question: Optional[str] = None
    chat_messages: Optional[List[Dict]] = None
    merged_content: Optional[str] = None
    error: Optional[str] = None


class QueryOpenAIRequest(BaseModel):
    """OpenAI payload request"""
    collection_name: str = Field(..., description="Collection name to query")
    query: str = Field(..., description="User question")
    k: int = Field(default=5, description="Number of top-k results to retrieve")
    stream: bool = Field(default=True, description="Whether to stream response")
    model: str = Field(default="gpt-4", description="Model name")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Additional parameters")


class QueryOpenAIResponse(BaseModel):
    """OpenAI payload response"""
    success: bool
    payload_raw: str = ""
    message: str = ""


class CollectionsResponse(BaseModel):
    """Available collections response"""
    collections: List[str]
    count: int


# In-memory task management
class TaskStatus:
    """Task status in memory"""
    def __init__(self, task_id: str, collection_name: str, file_count: int):
        self.task_id = task_id
        self.collection_name = collection_name
        self.file_count = file_count
        self.status = "pending"
        self.message = "Task created"
        self.created_at = datetime.utcnow()
        self.completed_at = None
        self.result = None
        self.error_message = None


# Create FastAPI app
app = FastAPI(
    title="Document Processing & KV Cache API",
    description="Intelligent document processing and KV Cache generation API",
    version="3.0.0"
)

# Initialize services (using configured base folder)
task_manager = TaskManagerService(base_folder=settings.BASE_FOLDER)
rag_query_service = RAGQueryService()

# In-memory task store
tasks: Dict[str, TaskStatus] = {}

# Token task store
token_tasks: Dict[str, dict] = {}

# API-level processing flag
_is_processing = False

# Create a lock
_processing_lock = threading.Lock()


@app.on_event("startup")
async def startup():
    """Startup initialization"""
    logger.info("Document Processing & KV Cache API started")


@app.on_event("shutdown")
async def shutdown():
    """Shutdown cleanup"""
    logger.info("API shutdown completed")


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "3.0.0",
        "base_folder": task_manager.base_folder
    }


@app.get("/api/v1/gpu/status")
async def get_gpu_status():
    """Get processing status"""
    global _is_processing
    
    return {
        "is_busy": _is_processing,
        "timestamp": datetime.utcnow()
    }


@app.post("/api/v1/process", response_model=TaskStatusResponse)
async def create_processing_task(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a document processing and KV Cache generation task
    
    Args:
        request: processing request
        background_tasks: background task handler
        
    Returns:
        task status
    """
    global _is_processing
    
    try:
        # Check if another request is being processed and set flag atomically
        with _processing_lock:
            if _is_processing:
                raise HTTPException(
                    status_code=409,  # Conflict
                    detail="System is processing another request, please retry later"
                )
            # Set processing flag
            _is_processing = True
        
        task_id = str(uuid.uuid4())
        logger.info(f"Received processing request: task_id={task_id}, collection={request.collection_name}, files={len(request.file_list)}")
        
        # Validate file info
        missing_files = []
        unsupported_files = []
        minio_files = []
        local_files = []
        
        for file_info in request.file_list:
            file_path = file_info.path
            filename = file_info.filename
            
            if file_path.startswith('s3://') or file_path.startswith('minio://') or file_path.startswith('http'):
                # Remote file; will use external parser supporting various formats
                minio_files.append(file_info)
            else:
                # Local file path - only .txt supported
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                elif not filename.lower().endswith('.txt'):
                    unsupported_files.append(filename)
                else:
                    local_files.append(file_info)
        
        if missing_files:
            raise HTTPException(
                status_code=400, 
                detail=f"Local files do not exist: {missing_files}"
            )
        
        if unsupported_files:
            raise HTTPException(
                status_code=400, 
                detail=f"Only .txt is supported for local files, unsupported: {unsupported_files}"
            )
        
        logger.info(f"Files - local: {len(local_files)}, remote/MinIO: {len(minio_files)}")
        
        # Create in-memory task status
        task_status = TaskStatus(task_id, request.collection_name, len(request.file_list))
        tasks[task_id] = task_status
        
        # Add background task
        background_tasks.add_task(
            process_documents_background,
            task_id=task_id,
            request=request,
            local_files_count=len(local_files),
            minio_files_count=len(minio_files)
        )
        
        return TaskStatusResponse(
            task_id=task_id,
            status="pending",
            message=f"Task created, start processing {len(request.file_list)} files"
        )
        
    except Exception as e:
        # Ensure processing flag is reset when error occurs
        with _processing_lock:
            _is_processing = False
        logger.error(f"Task creation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get task status
    
    Args:
        task_id: task ID
        
    Returns:
        task status
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task_status.status,
        message=task_status.error_message if task_status.error_message else task_status.message
    )

    
@app.post("/api/v1/tokens", response_model=TokenResponse)
async def calculate_tokens(
    request: TokenRequest,
    background_tasks: BackgroundTasks
):
    """
    calculate the token count of a single file
    
    Args:
        request: Token calculate request
        background_tasks: background task
        
    Returns:
        Token calculate result
    """
    try:
        task_id = str(uuid.uuid4())
        logger.info(f"Received token calculate request: task_id={task_id}, file={request.filename}")
        
        # Validate file format (remote files support multiple formats, local only supports txt)
        file_path = request.file_path
        filename = request.filename
        
        if not (file_path.startswith('s3://') or file_path.startswith('minio://') or 
                file_path.startswith('http://') or file_path.startswith('https://')):
            # Local file, only supports txt
            if not filename.lower().endswith('.txt'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Local file only supports .txt format, current file: {filename}"
                )
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Local file does not exist: {file_path}"
                )
        
        # Add background task
        background_tasks.add_task(
            process_token_calculation_background,
            task_id=task_id,
            file_path=file_path,
            filename=filename
        )
        
        # Initialize token calculation task status
        token_tasks[task_id] = {
            "status": "processing",
            "taskID": task_id,
            "fileName": filename,
            "token_count": 0,
            "message": "Token calculate task created, processing...",
            "error": None
        }
        
        return TokenResponse(
            status=True,
            taskID=task_id,
            fileName=filename,
            token_count=0,  # Will be updated after background calculation is complete
            message="Token calculate task created, processing..."
        )
        
    except Exception as e:
        logger.error(f"Failed to create token calculate task: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/tokens/status/{task_id}", response_model=TokenResponse)
async def get_token_task_status(task_id: str):
    """
    Get token calculate task status
    
    Args:
        task_id: task ID
        
    Returns:
        Token calculate result
    """
    # Record request start
    logger.info(f"[/api/v1/tokens/status/{task_id}] Received request - task_id: {task_id}")
    
    if task_id not in token_tasks:
        logger.warning(f"[/api/v1/tokens/status/{task_id}] Request failed - task does not exist: {task_id}")
        raise HTTPException(status_code=404, detail="Token calculate task does not exist")
    
    task_data = token_tasks[task_id]
    
    # Record found task data
    logger.info(f"[/api/v1/tokens/status/{task_id}] Found task data: {task_data}")
    
    response = TokenResponse(
        status=task_data["status"] == "completed",
        taskID=task_data["taskID"],
        fileName=task_data["fileName"],
        token_count=task_data["token_count"],
        message=task_data["message"],
        error=task_data.get("error")
    )
    
    # Record complete response content
    logger.info(f"[/api/v1/tokens/status/{task_id}] Response content: {response.model_dump()}")
    
    return response

@app.get("/api/v1/info")
async def get_api_info():
    """Get API information"""
    return {
        "api_name": "Document Processing & KV Cache API",
        "version": "3.0.0",
        "base_folder": task_manager.base_folder,
        "active_tasks": len(tasks),
        "endpoints": [
            "POST /api/v1/process - create task (supports language parameter)",
            "GET /api/v1/status/{task_id} - get task status", 
            "GET /api/v1/gpu/status - get GPU status",
            "GET /api/v1/info - API info",
            "GET /health - health check",
            "POST /api/v1/tokens - calculate single file token count",
            "GET /api/v1/tokens/status/{task_id} - query token calculate task status",
            "GET /api/v1/collections - get available collections",
            "POST /api/v1/query - RAG query with collection selection",
            "POST /api/v1/query/openai - generate OpenAI format payload"
        ],
        "supported_formats": {
            "local_files": ["txt"],
            "remote_files": ["pdf", "docx", "xlsx", "txt", "csv", "pptx", "html", "md"]
        }
    }


async def process_documents_background(
    task_id: str, 
    request: ProcessingRequest, 
    local_files_count: int = 0, 
    minio_files_count: int = 0,
):
    """
    Background document processing task
    
    Args:
        task_id: task ID
        request: processing request
        local_files_count: number of local files
        minio_files_count: number of MinIO/remote files
        language: language for prompt template
    """
    global _is_processing
    task_status = tasks[task_id]
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Start background processing task: {task_id}")
        
        # Update status
        task_status.status = "processing"
        task_status.message = "Processing documents and generating KV Cache..."
        
        # Use the new process_collection_workflow
        result = await task_manager.process_collection_workflow(
            collection_name=request.collection_name,
            input_files=request.file_list,
            language=request.language
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        if result.get("success", False):
            # Build path info
            collection_folder = os.path.join(task_manager.base_folder, request.collection_name)
            file_content_dir = os.path.join(collection_folder, "file_content")
            processed_output_dir = os.path.join(collection_folder, "processed_output")
            kvcache_dir = os.path.join(collection_folder, "kvcache")
            
            # Create success response
            processing_response = ProcessingResponse(
                task_id=task_id,
                status="completed",
                collection_name=request.collection_name,
                input_files_count=result.get("input_files_count", 0),
                documents_count=result.get("documents_count", 0),
                merged_files_count=len(result.get("merged_files", [])),
                kvcache_processed_count=result.get("kvcache_processed_count", 0),
                processing_time=processing_time,
                local_files_count=local_files_count,
                minio_files_count=minio_files_count,
                collection_folder=collection_folder,
                file_content_dir=file_content_dir,
                processed_output_dir=processed_output_dir,
                kvcache_dir=kvcache_dir,
                merged_files=result.get("merged_files", []),
                message="Processing completed"
            )
            
            # Update task status
            task_status.status = "completed"
            task_status.message = "Processing completed"
            task_status.completed_at = datetime.utcnow()
            task_status.result = processing_response
            
            logger.info(f"Background task completed: {task_id}")
        else:
            # Failure
            error_msg = result.get("error", "Unknown error")
            processing_response = ProcessingResponse(
                task_id=task_id,
                status="failed",
                collection_name=request.collection_name,
                input_files_count=result.get("input_files_count", 0),
                processing_time=processing_time,
                local_files_count=local_files_count,
                minio_files_count=minio_files_count,
                error=error_msg,
                message=f"Processing failed: {error_msg}"
            )
            
            task_status.status = "failed"
            task_status.error_message = error_msg
            task_status.message = f"Processing failed: {error_msg}"
            task_status.result = processing_response
        
    except Exception as e:
        logger.error(f"Background processing failed: {task_id}, error: {str(e)}")
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        processing_response = ProcessingResponse(
            task_id=task_id,
            status="failed",
            collection_name=request.collection_name,
            input_files_count=len(request.file_list),
            processing_time=processing_time,
            error=str(e),
            message=f"Processing failed: {str(e)}"
        )
        
        task_status.status = "failed"
        task_status.error_message = str(e)
        task_status.message = f"Processing failed: {str(e)}"
        task_status.result = processing_response
    
    finally:
        # Reset processing flag
        with _processing_lock:
            _is_processing = False
        logger.info(f"Reset processing flag; new requests allowed")


async def process_token_calculation_background(
    task_id: str,
    file_path: str,
    filename: str
):
    """
    Background processing of token calculation for a single file
    
    Args:
        task_id: task ID
        file_path: file path
        filename: file name
    """
    from services.external_parser import ExternalParserService
    from services.simple_txt_loader import SimpleTxtLoader
    from transformers import AutoTokenizer
    from pathlib import Path
    
    logger.info(f"Start background processing of token calculation: {task_id}, file: {filename}")
    
    try:
        # Create token output directory
        token_output_dir = f"/mnt/nvme0/cache/token/{task_id}"
        os.makedirs(token_output_dir, exist_ok=True)
        logger.info(f"Create output directory: {token_output_dir}")
        
        documents = []
        
        # According to file type, select processing method
        if (file_path.startswith('s3://') or file_path.startswith('minio://') or 
            file_path.startswith('http://') or file_path.startswith('https://')):
            # Remote file - use external parser
            logger.info(f"Use external parser to process remote file: {filename}")
            external_parser = ExternalParserService()
            
            # Create FileInfo mock object
            class FileInfo:
                def __init__(self, path, filename):
                    self.path = path
                    self.filename = filename
            
            file_info = FileInfo(file_path, filename)
            documents = await external_parser.parse_single_file_with_name(file_path, filename)
        else:
            # Local file - use simple txt loader
            logger.info(f"Use simple loader to process local txt file: {filename}")
            txt_loader = SimpleTxtLoader()
            doc = txt_loader._load_single_file_with_name(file_path, filename)
            documents = [doc] if doc else []
        
        if not documents:
            raise ValueError("No successful document content parsed")
        
        # Merge document content
        merged_content = ""
        for doc in documents:
            content = doc.page_content.strip()
            if content:
                merged_content += content + "\n\n"
        
        # Save merged txt file
        safe_filename = filename + ".txt"
        output_file_path = os.path.join(token_output_dir, safe_filename)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(merged_content)
        
        logger.info(f"Conversion completed, saved: {output_file_path} ({len(merged_content)} characters)")
        
        # Calculate token count
        tokenizer = AutoTokenizer.from_pretrained(settings.LLM_TOKENIZER_PATH)
        tokens = tokenizer(merged_content, return_tensors=None, add_special_tokens=False)["input_ids"]
        token_count = len(tokens)
        
        logger.info(f"Token calculation completed: {filename} = {token_count} tokens")
        
        # Update token calculation task success status
        if task_id in token_tasks:
            token_tasks[task_id].update({
                "status": "completed",
                "token_count": token_count,
                "message": f"Token calculation completed: {token_count} tokens"
            })
        
        logger.info(f"Token calculation task completed: task_id={task_id}, file={filename}, tokens={token_count}")
        
    except Exception as e:
        logger.error(f"Token calculation background processing failed: {task_id}, error: {str(e)}")
        
        # Update token calculation task error status
        if task_id in token_tasks:
            token_tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "message": f"Token calculation failed: {str(e)}"
            })


# ============================================================================
# RAG Query API Endpoints
# ============================================================================

@app.get("/api/v1/collections", response_model=CollectionsResponse)
async def get_available_collections():
    """
    獲取可用的 collection 列表
    
    Returns:
        CollectionsResponse: 包含所有可用 collection 的列表
    """
    logger.info("[/api/v1/collections] Received request to get available collections")
    
    try:
        collections = rag_query_service.get_available_collections()
        
        response = CollectionsResponse(
            collections=collections,
            count=len(collections)
        )
        
        logger.info(f"[/api/v1/collections] Found {len(collections)} collections: {collections}")
        return response
        
    except Exception as e:
        logger.error(f"[/api/v1/collections] Error getting collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting collections: {str(e)}")


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_collection(request: QueryRequest):
    """
    對指定的 collection 進行 RAG 查詢
    
    Args:
        request: QueryRequest 包含 collection_name, question, k 參數
    
    Returns:
        QueryResponse: 查詢結果，包含推薦的文件名、聊天消息和檢索內容
    """
    logger.info(f"[/api/v1/query] Received query request - collection: {request.collection_name}, question: {request.question[:50]}...")
    
    try:
        # 驗證 collection 是否存在
        available_collections = rag_query_service.get_available_collections()
        if request.collection_name not in available_collections:
            logger.warning(f"[/api/v1/query] Collection not found: {request.collection_name}")
            raise HTTPException(
                status_code=404, 
                detail=f"Collection '{request.collection_name}' not found. Available collections: {available_collections}"
            )
        
        # 執行 RAG 查詢
        # result: {'filename': merge_file_name,
        #          'chat_messages': chat_messages,
        #          'merged_content': merged_content,
        #          'error': ''}
        result = rag_query_service.get_rag_context_with_file_content(
            collection_name=request.collection_name,
            question=request.question,
            k=request.k
        )
        
        if result.get('error'):
            logger.error(f"[/api/v1/query] Query failed: {result['error']}")
            return QueryResponse(
                success=False,
                filename=None,
                file_path=None,
                question=request.question,
                chat_messages=[],
                merged_content="",
                error=result['error']
            )
        else:
            # 構建完整的文件路徑
            filename = result.get('filename', '')
            file_path = None
            if filename:
                # 嘗試多個可能的路徑來構建完整路徑
                possible_paths = [
                    os.path.join(settings.BASE_FOLDER, request.collection_name, "merged_files", filename),
                    os.path.join(settings.BASE_FOLDER, request.collection_name, "processed_output", "merged_files", filename),
                    os.path.join("tmp", request.collection_name, "processed_output", "merged_files", filename)
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        file_path = path
                        break
                
                # 如果找不到實際文件，使用第一個路徑作為默認值
                if file_path is None:
                    file_path = possible_paths[0]
            
            logger.info(f"[/api/v1/query] Query successful - filename: {filename}, file_path: {file_path}")
            return QueryResponse(
                success=True,
                filename=filename,
                file_path=file_path,
                question=request.question,
                chat_messages=result['chat_messages'],
                merged_content=result.get('merged_content', ''),
                error=""
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/api/v1/query] Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/v1/query/openai", response_model=QueryOpenAIResponse)
async def query_collection_openai_payload(request: QueryOpenAIRequest):
    """
    生成標準 OpenAI 格式的 payload
    
    Args:
        request: QueryOpenAIRequest 包含查詢參數和 OpenAI 配置
    
    Returns:
        QueryOpenAIResponse: 包含 OpenAI 格式的 payload 字符串
    """
    logger.info(f"[/api/v1/query/openai] Received OpenAI payload request - collection: {request.collection_name}, query: {request.query[:50]}...")
    
    try:
        # 驗證 collection 是否存在
        available_collections = rag_query_service.get_available_collections()
        if request.collection_name not in available_collections:
            logger.warning(f"[/api/v1/query/openai] Collection not found: {request.collection_name}")
            raise HTTPException(
                status_code=404, 
                detail=f"Collection '{request.collection_name}' not found. Available collections: {available_collections}"
            )
        
        # 生成 OpenAI payload
        result = rag_query_service.generate_openai_payload(
            collection_name=request.collection_name,
            query=request.query,
            k=request.k,
            stream=request.stream,
            model=request.model,
            params=request.params
        )
        
        if result['success']:
            payload_raw = result['payload_raw']
            messages = payload_raw['messages']


            logger.info(f"[/api/v1/query/openai] OpenAI payload generated successfully")
            return QueryOpenAIResponse(
                success=True,
                payload_raw=result['payload_raw'],
                message=result['message']
            )
        else:
            logger.error(f"[/api/v1/query/openai] Failed to generate OpenAI payload: {result['message']}")
            return QueryOpenAIResponse(
                success=False,
                payload_raw="",
                message=result['message']
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/api/v1/query/openai] Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG
    ) 
