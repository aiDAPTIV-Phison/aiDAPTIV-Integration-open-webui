"""
KV Cache generation service
Responsible for invoking external LLM tools to generate KV cache bin files
"""
import os
import asyncio
import subprocess
import time
import httpx
from typing import List, Dict, Any, Optional
from loguru import logger
from langchain_community.docstore.document import Document
import sys
import os
from datetime import datetime
# Ensure parent directory is in sys.path
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import settings



class KVCacheGeneratorService:
    """KV Cache generation service"""
    
    def __init__(self, collection_name: str = "default",
                       base_folder: str = "./data"):
        self.llm_api_url = settings.LLM_API_URL
        self.llm_api_key = settings.LLM_API_KEY
        self.collection_name = collection_name
        self.base_folder = base_folder
        
        # Collection folder path (created by task_manager)
        self.collection_folder = os.path.join(self.base_folder, self.collection_name)        
        self.client = httpx.AsyncClient()
        
        # Instance-level phisonai process
        self.phisonai_process = None
        
        # Build command using config parameters, offload path uses collection folder
        # Build command parameters
        cmd_parts = [
            settings.LLAMA_SERVER_PATH,
            "-m", settings.LLM_MODEL_PATH,
            "-n", str(settings.LLAMA_SERVER_N_PARALLEL),
            "-c", str(settings.LLM_CONTEXT_LENGTH),
            "-s", str(settings.LLAMA_SERVER_SEED),
            "--host", settings.LLAMA_SERVER_HOST,
            "-mg", str(settings.LLAMA_SERVER_MAIN_GPU),
            "-ngl", str(settings.LLAMA_SERVER_N_GPU_LAYERS),
            "--offload-path", self.collection_folder,
            "--ssd-kv-offload-gb", str(settings.LLAMA_SERVER_SSD_KV_OFFLOAD_GB),
            "--dram-kv-offload-gb", str(settings.LLAMA_SERVER_DRAM_KV_OFFLOAD_GB),
            "--kv-cache-resume-policy", str(settings.LLAMA_SERVER_KV_CACHE_RESUME_POLICY),
            "--slot-save-path", self.collection_folder,
            "--port", str(settings.LLM_API_PORT)
        ]
        
        # Append boolean flags
        if settings.LLAMA_SERVER_FLASH_ATTN:
            cmd_parts.append("--flash-attn")
        if settings.LLAMA_SERVER_NO_CONTEXT_SHIFT:
            cmd_parts.append("--no-context-shift")
        
        self.phisonai_command = " ".join(cmd_parts)
        
        # Environment variables
        # Derive llama-server directory for LD_LIBRARY_PATH
        llama_server_dir = os.path.dirname(settings.LLAMA_SERVER_PATH)
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        
        # Ensure llama-server directory is in LD_LIBRARY_PATH
        if current_ld_path:
            new_ld_path = f"{llama_server_dir}:{current_ld_path}"
        else:
            new_ld_path = llama_server_dir
            
        self.env_vars = {
            "CUDA_VISIBLE_DEVICES": str(settings.LLAMA_SERVER_MAIN_GPU),
            "LD_LIBRARY_PATH": new_ld_path
        }
        
        self.phisonai_working_dir = "./"
        self.startup_timeout = 60
        self.health_check_interval = 5.0
        self.health_check_prompt = "hi"
        self.health_check_max_tokens = 3
        self.health_check_timeout = 5.0
        self.request_timeout = 3600
        self.cache_format = "bin"
        
        logger.info(f"KVCacheGeneratorService initialized: API={self.llm_api_url}")
        logger.info(f"KV cache path: {self.collection_folder}")
        logger.info(f"Working directory: {self.phisonai_working_dir}")
    
    def get_status(self) -> dict:
        """Get current status information"""
        return {
            "is_running": self.phisonai_process is not None and self.phisonai_process.poll() is None,
            "current_collection": self.collection_name,
            "phisonai_pid": self.phisonai_process.pid if self.phisonai_process else None
        }
    
    async def generate_kvcaches_for_collection(self, collection_name: str, merged_files: List[str], language: str = "zh-TW") -> int:
        """
        Generate KV cache for all merged files of a collection
        Start PhisonAI only once for the collection, then process all files
        
        Args:
            collection_name: Collection name
            merged_files: List of merged file paths
            language: Language for prompt template (zh-TW, en, english)
            
        Returns:
            Number of files processed successfully
        """
        try:
            logger.info(f"Start generating KV cache for collection '{collection_name}', files: {len(merged_files)}")
            
            # Directory already created during initialization, no need to create again
            
            # Start PhisonAI service (start only once) - this will acquire process lock
            logger.info("Starting PhisonAI service...")
            await self._start_phisonai_with_lock()
            
            try:
                processed_count = 0
                
                # Use for loop to process each merged file
                for i, file_path in enumerate(merged_files):
                    logger.info(f"Processing {i+1}/{len(merged_files)}: {os.path.basename(file_path)}")
                    
                    try:
                        # Read merged file content
                        merge_content = self._load_merged_file(file_path)
                        
                        # Extract original filename (strip .txt)
                        full_filename = os.path.basename(file_path)
                        if full_filename.endswith('.txt'):
                            original_filename = full_filename[:-4]
                        else:
                            original_filename = full_filename
                        
                        task_id = f"{collection_name}_{original_filename}"
                        
                        # Generate KV cache (no restart); auto-saved by phisonai
                        await self._generate_single_kvcache(merge_content, task_id, language)
                        processed_count += 1
                        
                        logger.info(f"✓ KV cache generated for '{original_filename}'")
                        
                    except Exception as e:
                        logger.error(f"✗ Failed to process file {file_path}: {str(e)}")
                        continue
                
                logger.info(f"Collection '{collection_name}' KV cache done: {processed_count}/{len(merged_files)} succeeded")
                return processed_count
                
            finally:
                # After collection processing, close PhisonAI process and release lock
                await self._stop_phisonai_with_lock()
            
        except Exception as e:
            logger.error(f"Collection '{collection_name}' KV cache generation failed: {str(e)}")
            # Ensure process is closed and lock is released even in exception cases
            await self._stop_phisonai_with_lock()
            raise
    
    async def generate_kvcache(self, merge_content: List[Document], task_id: str, output_path: str) -> str:
        """
        Generate single KV cache bin file (legacy interface, for compatibility)
        
        Args:
            merge_content: Merged document content list, containing page_content and metadata
            task_id: Task ID
            output_path: Output path
            
        Returns:
            Generated KV cache file path
        """
        try:
            logger.info(f"Start generating KV cache: task_id={task_id}")
            
            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)
            
            # Start PhisonAI service and acquire lock
            await self._start_phisonai_with_lock()
            
            try:
                # Generate KV cache
                kvcache_path = await self._generate_single_kvcache(merge_content, task_id)
                
                logger.info(f"KV cache generation completed: {kvcache_path}")
                return kvcache_path
                
            finally:
                # After processing, close PhisonAI process and release lock
                await self._stop_phisonai_with_lock()
            
        except Exception as e:
            logger.error(f"KV cache generation failed: task_id={task_id}, error: {str(e)}")
            # Ensure process is closed and lock is released even in exception cases
            await self._stop_phisonai_with_lock()
            raise
    
    async def _terminate_existing_phisonai(self):
        """Terminate existing phisonai process"""
        if self.phisonai_process and self.phisonai_process.poll() is None:
            logger.info(f"Terminating existing phisonai process (PID: {self.phisonai_process.pid})...")
            
            try:
                # Try graceful termination
                self.phisonai_process.terminate()
                
                # Wait up to 2000 seconds
                for i in range(2000):
                    await asyncio.sleep(1)
                    if self.phisonai_process.poll() is not None:
                        logger.info(f"phisonai process terminated gracefully (PID: {self.phisonai_process.pid}), waited {i+1} seconds")
                        break
                else:
                    # Force kill if graceful termination fails
                    logger.warning(f"Graceful termination failed; forcing kill for phisonai (PID: {self.phisonai_process.pid})...")
                    self.phisonai_process.kill()
                    
                    # Wait after forced kill
                    for i in range(5):
                        await asyncio.sleep(1)
                        if self.phisonai_process.poll() is not None:
                            logger.info(f"phisonai process force-killed (PID: {self.phisonai_process.pid}), waited {i+1} seconds")
                            break
                    else:
                        logger.error(f"Unable to terminate phisonai process (PID: {self.phisonai_process.pid})")
                        
            except Exception as e:
                logger.error(f"Error terminating phisonai process: {str(e)}")
                
        elif self.phisonai_process:
            logger.info(f"phisonai process already exited (PID: {self.phisonai_process.pid}, exit code: {self.phisonai_process.poll()})")
        else:
            logger.info("No running phisonai process")
    
    async def _start_phisonai_process(self):
        """Start phisonai process"""
        logger.info(f"Starting phisonai process...")
        logger.info(f"Model path: {settings.LLM_MODEL_PATH}")
        logger.info(f"KV cache directory: {self.collection_folder}")
        
        # Use pre-built command
        command_parts = self.phisonai_command.split()
        logger.info(f"Executing command: {self.phisonai_command}")
        
        # Set environment variables
        env = os.environ.copy()
        env.update(self.env_vars)
        logger.info(f"phison ai command: {command_parts}")
        
        # ===== New: Create log file path (with timestamp) =====
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"llama_server_{timestamp}.log"
        
        # Use agent_builder_km_HP/logs/ directory
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        log_file_path = os.path.join(logs_dir, log_filename)
        
        # Ensure log directory exists
        os.makedirs(logs_dir, exist_ok=True)
        
        logger.info(f"llama-server log will be saved to: {log_file_path}")
        # ===== New section end =====
        
        try:
            # Start process, redirect output to log file (minimal necessary change)
            log_fh = open(log_file_path, 'a', encoding='utf-8')
            self.phisonai_process = subprocess.Popen(
                command_parts,
                stdout=log_fh,
                stderr=log_fh,
                env=env,
                cwd=self.phisonai_working_dir,
                bufsize=1,
                universal_newlines=True
            )
            logger.info(f"phisonai started (PID: {self.phisonai_process.pid}) with CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
            logger.info(f"Log file: {log_file_path}")
            
            # Brief check if process exits immediately
            time.sleep(2)
            if self.phisonai_process.poll() is not None:
                exit_code = self.phisonai_process.poll()
                logger.error(f"Process startup failed, exit code: {exit_code}")
                try:
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        logger.error(f"Log content:\n{f.read()}")
                except Exception:
                    pass
                raise Exception(f"phisonai process startup failed, exit code: {exit_code}")
            
        except FileNotFoundError as e:
            logger.error(f"llama-server executable not found: {settings.LLAMA_SERVER_PATH}")
            logger.error(f"Please check if path is correct and file exists with execute permissions")
            raise Exception(f"llama-server file does not exist: {e}")
        except PermissionError as e:
            logger.error(f"No permission to execute llama-server: {e}")
            logger.error(f"Please check file permissions: chmod +x {settings.LLAMA_SERVER_PATH}")
            raise Exception(f"Permission error: {e}")
        except Exception as e:
            logger.error(f"Error starting phisonai process: {e}")
            raise Exception(f"Process startup failed: {e}")

    def _get_exit_code_meaning(self, exit_code: int) -> str:
        """Get exit code meaning"""
        exit_codes = {
            0: "Normal exit",
            1: "General error",
            2: "Shell command misuse",
            126: "Command cannot execute",
            127: "Command not found",
            128: "Invalid exit argument",
            130: "Terminated by Ctrl+C",
            137: "Terminated by SIGKILL signal",
            139: "Segmentation fault (SIGSEGV)",
            143: "Terminated by SIGTERM signal"
        }
        return exit_codes.get(exit_code, f"Unknown exit code ({exit_code})")

    def _analyze_exit_reason(self, exit_code: int, stdout: str, stderr: str):
        """Analyze process exit reason"""
        logger.error("=== Process Exit Reason Analysis ===")
        
        # Check common error patterns
        if exit_code == 137:
            logger.error("Possible cause: Out of memory killed by system (OOM Killer)")
        elif exit_code == 139:
            logger.error("Possible cause: Segmentation fault, possibly memory access error or corrupted model file")
        elif exit_code == 1:
            logger.error("Possible cause: General error, check the following common issues:")
            
            error_text = (stdout + stderr).lower()
            
            if "cuda" in error_text:
                logger.error("  - CUDA related error: Check GPU driver and CUDA version")
            if "memory" in error_text or "out of memory" in error_text:
                logger.error("  - Memory insufficient: Reduce model size or increase system memory")
            if "permission" in error_text:
                logger.error("  - Permission error: Check file and directory permissions")
            if "not found" in error_text:
                logger.error("  - File not found: Check model file path")
            if "invalid" in error_text:
                logger.error("  - Invalid parameters: Check command line arguments")
                
        logger.error("========================")

    def _analyze_startup_failure(self, exit_code: int, stdout_lines: list, stderr: str):
        """Analyze startup failure reason"""
        logger.error("=== Startup Failure Analysis ===")
        logger.error(f"Exit code: {exit_code} ({self._get_exit_code_meaning(exit_code)})")
        
        # Analyze stdout
        if stdout_lines:
            logger.error("Key information in stdout:")
            for line in stdout_lines:
                if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'abort', 'warning']):
                    logger.error(f"  {line}")
        
        # Analyze stderr
        if stderr:
            logger.error(f"Error output:\n{stderr}")
        
        # Check common issues
        all_output = "\n".join(stdout_lines) + "\n" + stderr
        self._check_common_issues(all_output)
        
        logger.error("========================")

    def _check_common_issues(self, output: str):
        """Check common issues"""
        output_lower = output.lower()
        
        logger.error("Common issues check:")
        
        if "model" in output_lower and ("not found" in output_lower or "no such file" in output_lower):
            logger.error("  ❌ Model file does not exist or path error")
            logger.error(f"     Please check model path: {settings.LLM_MODEL_PATH}")
            
        if "cuda" in output_lower:
            if "out of memory" in output_lower:
                logger.error("  ❌ GPU memory insufficient")
                logger.error("     Please try reducing -ngl parameter or use smaller model")
            elif "driver" in output_lower:
                logger.error("  ❌ CUDA driver issue")
                logger.error("     Please check GPU driver and CUDA version")
            else:
                logger.error("  ⚠️ CUDA related issue")
                
        if "permission" in output_lower:
            logger.error("  ❌ Permission issue")
            logger.error("     Please check file and directory permissions")
            
        if "port" in output_lower and ("bind" in output_lower or "already in use" in output_lower):
            logger.error(f"  ❌ Port {settings.LLM_API_PORT} already in use")
            logger.error("     Please check if other processes are using this port")
            
        if "invalid" in output_lower and "argument" in output_lower:
            logger.error("  ❌ Invalid command line arguments")
            logger.error("     Please check llama-server version and parameter compatibility")

    async def _wait_for_service_ready(self):
        """Wait for service to be ready"""
        logger.info("Waiting for phisonai service to be ready...")
        max_attempts = self.startup_timeout
        
        for attempt in range(max_attempts):
            try:
                # Check process status
                if self.phisonai_process:
                    poll_result = self.phisonai_process.poll()
                    if poll_result is not None:
                        # Process has exited, get output immediately
                        try:
                            # Read remaining output
                            remaining_stdout = self.phisonai_process.stdout.read() if self.phisonai_process.stdout else ""
                            remaining_stderr = self.phisonai_process.stderr.read() if self.phisonai_process.stderr else ""
                            
                            logger.error(f"phisonai process exited unexpectedly (exit code: {poll_result})")
                            logger.error(f"Exit code meaning: {self._get_exit_code_meaning(poll_result)}")
                            
                            if remaining_stdout:
                                logger.error(f"Last STDOUT output:\n{remaining_stdout}")
                            if remaining_stderr:
                                logger.error(f"Last STDERR output:\n{remaining_stderr}")
                                
                            # Check common error causes
                            self._analyze_exit_reason(poll_result, remaining_stdout, remaining_stderr)
                            
                        except Exception as read_error:
                            logger.error(f"Error reading process output: {read_error}")
                        
                        raise Exception(f"phisonai process exited unexpectedly, exit code: {poll_result}")
                    else:
                        logger.info(f"Check {attempt + 1}: phisonai process running (PID: {self.phisonai_process.pid})")
                
                is_ready = await self._check_phisonai_ready()
                if is_ready:
                    logger.info(f"phisonai service ready, took {attempt + 1} seconds")
                    return
            except Exception as e:
                logger.info(f"Check {attempt + 1} failed: {str(e)}")
            
            await asyncio.sleep(self.health_check_interval)
        
        # After timeout, get final process status and output
        if self.phisonai_process:
            try:
                poll_result = self.phisonai_process.poll()
                if poll_result is not None:
                    logger.error(f"Process exited, exit code: {poll_result}")
                else:
                    logger.error("Process still running but service not ready")
                    # Terminate process
                    self.phisonai_process.terminate()
                    
                # Try to get output
                try:
                    stdout, stderr = self.phisonai_process.communicate(timeout=5)
                    if stdout:
                        logger.error(f"Timeout STDOUT:\n{stdout}")
                    if stderr:
                        logger.error(f"Timeout STDERR:\n{stderr}")
                except Exception as comm_error:
                    logger.error(f"Error getting timeout output: {comm_error}")
                    
            except Exception as timeout_error:
                logger.error(f"Error handling timeout state: {timeout_error}")
        
        raise Exception(f"phisonai service startup timeout ({max_attempts} seconds)")

    async def _check_phisonai_ready(self) -> bool:
        """
        Check if phisonai is ready
        Use simple 'hi' prompt and limit returned tokens to 3
        
        Returns:
            Whether ready
        """
        try:
            # Use standard chat completions format
            test_request = {
                "messages": [
                    {
                        "role": "user",
                        "content": self.health_check_prompt
                    }
                ],
                "max_tokens": self.health_check_max_tokens,
                "temperature": 0,
                "stream": False
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # If API key exists, add authentication header
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            response = await self.client.post(
                self.llm_api_url,
                json=test_request,
                headers=headers,
                timeout=self.health_check_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                # Check if choices field and content exist
                if result and result.get("choices") and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    if content:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _load_merged_file(self, file_path: str) -> List[Document]:
        """
        Load content from merged file as Document list
        
        Args:
            file_path: Merged file path
            
        Returns:
            Document list
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Create Document object
            document = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "file_name": os.path.basename(file_path)
                }
            )
            
            return [document]
            
        except Exception as e:
            logger.error(f"Failed to load merged file: {file_path}, error: {str(e)}")
            raise
    
    async def _generate_single_kvcache(self, merge_content: List[Document], task_id: str, language: str = "zh-TW") -> None:
        """
        Generate KV cache for single merged content (no service restart)
        PhisonAI will automatically save KV cache to specified directory
        
        Args:
            merge_content: Merged document content list
            task_id: Task ID
            language: Language for prompt template (zh-TW, en, english)
        """
        try:
            # Prepare data
            documents_data = self._prepare_documents_for_llm(merge_content)
            
            # Call LLM API, let PhisonAI automatically handle KV cache saving
            await self._call_llm_api(documents_data, task_id, language)
            
        except Exception as e:
            logger.error(f"Failed to generate single KV cache: task_id={task_id}, error: {str(e)}")
            raise
    
    def _prepare_documents_for_llm(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Prepare document data for LLM API (simplified: directly extract page_content)
        
        Args:
            documents: Document list, containing page_content and metadata
            
        Returns:
            Prepared data
        """
        prepared_data = []
        for doc in documents:
            prepared_data.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            })
        
        return prepared_data
    
    async def _call_llm_api(self, documents_data: List[Dict[str, Any]], task_id: str, language: str = "zh-TW") -> None:
        """
        Call external LLM API to generate KV cache
        Use standard chat completions format to call LLM
        PhisonAI will automatically save generated KV cache to directory specified by --slot-save-path
        
        Args:
            documents_data: Document data
            task_id: Task ID
            language: Language for prompt template (zh-TW, en, english)
        """
        try:
            logger.info(f"Start generating KV cache for task {task_id}")
            
            # Directly extract merged file content
            merged_content = ""
            for doc in documents_data:
                merged_content += doc['page_content']
            
            # Build user message content using language-specific prompt template
            from config import get_user_prompt_template
            logger.info(f"Using language: {language}")
            user_prompt_template = get_user_prompt_template(language)
            user_content = user_prompt_template.format(chunk=merged_content)
            
            # Build messages array
            messages = []
            
            # If SYSTEM_PROMPT is not empty, add system message
            if settings.SYSTEM_PROMPT and settings.SYSTEM_PROMPT.strip():
                messages.append({
                    "role": "system",
                    "content": settings.SYSTEM_PROMPT
                })
            
            # Add user message
            messages.append({
                "role": "system",
                "content": user_content
            })
            
            # logger.info(f"@@@ messages: {messages}")
            
            # Use standard chat completions format to call LLM
            request_data = {
                "messages": messages,
                "max_tokens": 2,  # Limit output token count
                "temperature": 0,  # Set to 0 for consistency
                "stream": False,   # Don't use streaming output
                "cache_prompt": True  # Enable KV cache functionality
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # If API key exists, add authentication header
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            logger.info(f"Sending request to LLM API, content length: {len(merged_content)} characters")
            logger.info(f"Using {len(messages)} messages, includes system prompt: {bool(settings.SYSTEM_PROMPT and settings.SYSTEM_PROMPT.strip())}")
            
            # Send request
            response = await self.client.post(
                self.llm_api_url,
                json=request_data,
                headers=headers,
                timeout=self.request_timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"LLM API response successful: {result.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")
            logger.info(f"KV cache automatically saved to: {self.collection_folder}")
            
        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            raise
    
    async def validate_kvcache_file(self, file_path: str) -> bool:
        """
        Validate if KV cache file is valid
        
        Args:
            file_path: File path
            
        Returns:
            Whether valid
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False
            
            # TODO: Add more validation logic
            return True
            
        except Exception:
            return False
    
    async def _start_phisonai_with_lock(self):
        """
        Check GPU status and start PhisonAI service
        Return error if GPU is being used
        """
        try:
            logger.info(f"Starting phisonai (Collection: {self.collection_name})...")
            
            # Terminate existing process (if exists)
            await self._terminate_existing_phisonai()
            
            # Start new phisonai process
            await self._start_phisonai_process()
            
            # Wait for service to be ready
            await self._wait_for_service_ready()
            
            logger.info(f"PhisonAI startup completed (Collection: {self.collection_name})")
            
        except Exception as e:
            raise e
                
        except Exception as e:
            logger.error(f"Failed to start phisonai (Collection: {self.collection_name}): {str(e)}")
            raise
    
    async def _stop_phisonai_with_lock(self):
        """
        Stop PhisonAI service
        """
        try:
            logger.info(f"Stopping PhisonAI (Collection: {self.collection_name})...")
            
            # Terminate PhisonAI process
            await self._terminate_existing_phisonai()
            
            logger.info(f"PhisonAI stopped (Collection: {self.collection_name})")
                
        except Exception as e:
            logger.error(f"Failed to stop phisonai (Collection: {self.collection_name}): {str(e)}")
            raise
    
    
    async def close(self):
        """Close client connection and terminate phisonai process"""
        # Terminate phisonai process
        if self.phisonai_process and self.phisonai_process.poll() is None:
            logger.info("Closing phisonai process...")
            self.phisonai_process.terminate()
            try:
                self.phisonai_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("phisonai process could not terminate normally, force killing")
                self.phisonai_process.kill()
        
        # Close HTTP client
        await self.client.aclose()
        
        logger.info("KVCacheGeneratorService resource cleanup completed") 
