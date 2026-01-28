# -*- coding: utf-8 -*-
"""
Simplified API - Document processing and KV Cache generation
"""
import os
import sys
import uuid
import threading
import json
import httpx
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from config import settings
from services.task_manager import TaskManagerService
from services.rag_query_service import RAGQueryService
from loguru import logger

# Import embedding models
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    EMBEDDING_MODELS_AVAILABLE = True
except ImportError:
    EMBEDDING_MODELS_AVAILABLE = False
    logger.warning("Embedding models not available. Install with: pip install langchain-openai langchain-community")


# Logging configuration directory
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


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
    language: str = Field(default="zh-TW", description="Language")


class QueryOpenAIResponse(BaseModel):
    """OpenAI payload response"""
    success: bool
    payload_raw: str = ""
    message: str = ""
    merged_file: Optional[str] = None
    source_files: Optional[List[str]] = None
    retrieved_chunks: Optional[List[str]] = None
    merged_content: Optional[str] = None


class QueryExecuteRequest(BaseModel):
    """執行查詢並取得模型回應的 request"""
    collection_name: str = Field(..., description="Collection name to query")
    query: str = Field(..., description="User question")
    k: int = Field(default=5, description="Number of top-k results to retrieve")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Additional parameters")
    language: str = Field(default="zh-TW", description="Language")
    model_url: Optional[str] = Field(default=None, description="自訂模型 API URL（若未提供則使用環境變數）")
    model_name: Optional[str] = Field(default=None, description="自訂模型名稱（若未提供則使用環境變數）")


class QueryExecuteResponse(BaseModel):
    """執行查詢並取得模型回應的 response"""
    success: bool
    model_response: str = ""
    message: str = ""
    merged_file: Optional[str] = None


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

# 全域變數：延遲初始化
embedding_model = None
rag_query_service = None

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
    global embedding_model, rag_query_service
    
    # 配置 logger（只在 startup 時執行一次）
    logger.remove()  # 移除所有預設 handler
    
    # 添加 console handler
    logger.add(
        sys.stdout,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm} | {level:8} | {name}:{function}:{line} - {message}",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    
    # 添加 file handler（使用時間戳記）
    logger.add(
        os.path.join(LOG_DIR, "app_{time:YYYYMMDD_HHmmss}.log"),
        rotation="10 MB",        # 檔案大小達 10MB 自動輪替
        retention="10 days",     # 保留 10 天的 log
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm} | {level:8} | {name}:{function}:{line} - {message}",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    
    logger.info("=" * 80)
    logger.info("Document Processing & KV Cache API started")
    logger.info("=" * 80)
    
    # 初始化 embedding_model
    if EMBEDDING_MODELS_AVAILABLE and settings.EMBEDDING_URL:
        try:
            embedding_url = settings.EMBEDDING_API_URL
            
            if settings.EMBEDDING_TYPE == "tei":
                embedding_model = HuggingFaceInferenceAPIEmbeddings(
                    api_url=embedding_url, 
                    api_key="empty"
                )
                logger.info(f"✅ 使用 TEI 嵌入模型: {embedding_url}")
            elif settings.EMBEDDING_TYPE in ["vllm", "openai", "llamacpp"]:
                embedding_model = OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL_NAME,
                    base_url=embedding_url,
                    api_key="EMPTY",
                    tiktoken_enabled=False,
                    check_embedding_ctx_length=False,
                    encoding_format="float"
                )
                logger.info(f"✅ 使用 OpenAI format 嵌入模型: {embedding_url}, model: {settings.EMBEDDING_MODEL_NAME}")
            else:
                logger.warning(f"不支援的 EMBEDDING_TYPE: {settings.EMBEDDING_TYPE}，將使用 BM25")
        except Exception as e:
            logger.warning(f"⚠️  無法創建 embedding_model: {e}，將使用 BM25")
            embedding_model = None
    else:
        if not EMBEDDING_MODELS_AVAILABLE:
            logger.warning("⚠️  Embedding 模組不可用，將使用 BM25")
        elif not settings.EMBEDDING_URL:
            logger.warning("⚠️  未設定 EMBEDDING_URL，將使用 BM25")
    
    # 初始化 RAG query service
    rag_query_service = RAGQueryService(embedding_model=embedding_model)
    logger.info("✅ RAG Query Service initialized")


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
        
        # 獲取 collection
        chroma = rag_query_service._get_collection(request.collection_name)
        
        # 執行 RAG 查詢
        # result: {'filename': merge_file_name,
        #          'chat_messages': chat_messages,
        #          'merged_content': merged_content,
        #          'error': ''}
        result = rag_query_service.get_rag_context_with_file_content(
            chroma=chroma,
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
        # available_collections = rag_query_service.get_available_collections()
        # if request.collection_name not in available_collections:
        #     logger.warning(f"[/api/v1/query/openai] Collection not found: {request.collection_name}")
        #     raise HTTPException(
        #         status_code=404, 
        #         detail=f"Collection '{request.collection_name}' not found. Available collections: {available_collections}"
        #     )
        
        # 生成 OpenAI payload
        result = rag_query_service.generate_openai_payload(
            collection_name=request.collection_name,
            query=request.query,
            k=request.k,
            stream=request.stream,
            model=request.model,
            params=request.params,
            language=request.language
        )
        
        if result['success']:
            logger.info(f"[/api/v1/query/openai] OpenAI payload generated successfully")
            return QueryOpenAIResponse(
                success=True,
                payload_raw=result['payload_raw'],
                message=result['message'],
                merged_file=result.get('merged_file'),
                source_files=result.get('source_files'),
                retrieved_chunks=result.get('retrieved_chunks'),
                merged_content=result.get('merged_content')
            )
        else:
            logger.error(f"[/api/v1/query/openai] Failed to generate OpenAI payload: {result['message']}")
            return QueryOpenAIResponse(
                success=False,
                payload_raw="",
                message=result['message'],
                merged_file=None,
                source_files=None,
                retrieved_chunks=None,
                merged_content=None
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/api/v1/query/openai] Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/v1/query/execute", response_model=QueryExecuteResponse)
async def execute_query_with_model(request: QueryExecuteRequest):
    """
    執行 RAG 查詢並直接調用模型取得回應
    
    Args:
        request: QueryExecuteRequest 包含查詢參數和 OpenAI 配置
    
    Returns:
        QueryExecuteResponse: 包含模型的回應結果
    """
    logger.info(f"[/api/v1/query/execute] Received execute request - collection: {request.collection_name}, query: {request.query[:50]}...")
    
    try:
        # 使用 prepare_rag_messages 準備 RAG messages（不需要序列化/反序列化）
        rag_result = rag_query_service.prepare_rag_messages(
            collection_name=request.collection_name,
            query=request.query,
            k=request.k,
            language=request.language
        )
        
        if not rag_result['success']:
            logger.error(f"[/api/v1/query/execute] Failed to prepare RAG messages: {rag_result['message']}")
            return QueryExecuteResponse(
                success=False,
                model_response="",
                message=rag_result['message'],
                merged_file=None,
                retrieved_chunks=None
            )
        
        # 調用 LLM API
        try:
            # 判斷使用自訂參數或環境變數（必須兩個都提供才使用自訂，否則都用預設）
            if request.model_url and request.model_name:
                use_model_url = request.model_url
                use_model_name = request.model_name
                logger.info(f"[/api/v1/query/execute] Using custom model settings")
            else:
                use_model_url = settings.LLM_API_URL
                use_model_name = settings.LLM_MODEL_NAME
                logger.info(f"[/api/v1/query/execute] Using default model settings from env")
            
            logger.info(f"[/api/v1/query/execute] Calling LLM API...")
            logger.info(f"[/api/v1/query/execute] Model URL: {use_model_url}")
            logger.info(f"[/api/v1/query/execute] Model Name: {use_model_name}")
            
            # 構建 LLM API payload（直接使用 messages，無需序列化/反序列化）
            llm_payload = {
                "model": use_model_name,
                "messages": rag_result['messages'],
                "max_tokens": request.params.get("max_tokens", 2048) if request.params else 2048,
                "temperature": request.params.get("temperature", 0.7) if request.params else 0.7,
                "top_p": request.params.get("top_p", 1.0) if request.params else 1.0,
                "stream": False,
            }
            
            # 調用 LLM API
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    use_model_url,
                    json=llm_payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {settings.LLM_API_KEY}" if settings.LLM_API_KEY else ""
                    }
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # 提取模型回應
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        model_response = response_data['choices'][0].get('message', {}).get('content', '')
                        logger.info(f"[/api/v1/query/execute] Model response received successfully")
                        
                        return QueryExecuteResponse(
                            success=True,
                            model_response=model_response,
                            message="Query executed successfully",
                            merged_file=rag_result.get('merged_file')
                        )
                    else:
                        model_response = json.dumps(response_data, ensure_ascii=False)
                        logger.warning(f"[/api/v1/query/execute] Unexpected response format")
                        
                        return QueryExecuteResponse(
                            success=False,
                            model_response=model_response,
                            message="Unexpected response format from LLM",
                            merged_file=rag_result.get('merged_file'),
                            retrieved_chunks=rag_result.get('retrieved_chunks')
                        )
                else:
                    error_msg = f"LLM API error: {response.status_code} - {response.text}"
                    logger.error(f"[/api/v1/query/execute] {error_msg}")
                    
                    return QueryExecuteResponse(
                        success=False,
                        model_response="",
                        message=error_msg,
                        merged_file=rag_result.get('merged_file'),
                        retrieved_chunks=rag_result.get('retrieved_chunks')
                    )
                    
        except Exception as e:
            error_msg = f"Failed to call LLM API: {str(e)}"
            logger.error(f"[/api/v1/query/execute] {error_msg}")
            
            return QueryExecuteResponse(
                success=False,
                model_response="",
                message=error_msg,
                merged_file=rag_result.get('merged_file'),
                retrieved_chunks=rag_result.get('retrieved_chunks')
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/api/v1/query/execute] Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG
    ) 
