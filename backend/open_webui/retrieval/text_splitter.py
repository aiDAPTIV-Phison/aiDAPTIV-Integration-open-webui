"""
Custom Text Splitter using Chat Model Endpoint for Token Counting
Supports LlamaCPP and VLLM tokenizers via OpenAI-compatible API

Usage:
    from open_webui.retrieval.text_splitter import APITokenTextSplitter
    
    # Basic usage
    splitter = APITokenTextSplitter(
        api_base_url="http://localhost:8080/v1",
        model_name="llama-3.1-8b",
        chunk_size=512,
        chunk_overlap=50
    )
    
    # With KV Cache prefilling using RAG template (same as middleware.py)
    splitter = APITokenTextSplitter(
        api_base_url="http://localhost:8080/v1",
        model_name="llama-3.1-8b",
        chunk_size=512,
        chunk_overlap=50,
        enable_kv_cache_prefill=True,
        rag_template="Context: {{CONTEXT}}\nQuery: {{QUERY}}",  # Uses rag_template like middleware.py
        prefill_query="Please read and understand this content."  # Placeholder query for prefill
    )
    
    # With custom payload builder (advanced usage)
    def custom_payload_builder(model_name: str, content: str) -> dict:
        '''Build payload similar to process_chat_payload in middleware.py'''
        return {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Custom system prompt"},
                {"role": "user", "content": content}
            ],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
            "metadata": {"task": "kv_cache_prefill"}
        }
    
    splitter = APITokenTextSplitter(
        api_base_url="http://localhost:8080/v1",
        model_name="llama-3.1-8b",
        chunk_size=512,
        chunk_overlap=50,
        enable_kv_cache_prefill=True,
        prefill_payload_builder=custom_payload_builder
    )
"""
import logging
import httpx
from typing import List, Optional, Callable
from langchain_core.documents import Document
from langchain.text_splitter import TextSplitter

# Import RAG template utilities from Open WebUI (same as middleware.py)
from open_webui.utils.task import rag_template
from open_webui.utils.misc import add_or_update_system_message

log = logging.getLogger(__name__)


class APITokenTextSplitter(TextSplitter):
    """
    Text splitter that uses a chat model endpoint to count tokens.
    Compatible with LlamaCPP and VLLM through OpenAI-compatible API.
    
    Args:
        api_base_url: Base URL of the API endpoint (e.g., "http://localhost:8000/v1")
        api_key: API key for authentication (optional)
        model_name: Model name to use for tokenization
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        timeout: Request timeout in seconds (default: 30)
        enable_kv_cache_prefill: Enable KV Cache prefilling by sending documents to LLM before returning (default: False)
        rag_template_str: RAG template string (same format as middleware.py), uses {{CONTEXT}} and {{QUERY}} placeholders (optional)
        prefill_query: Query placeholder to use during prefill (default: "Please understand this content")
        prefill_payload_builder: Custom function to build prefill payload, signature: (model_name: str, content: str) -> dict (optional)
    """

    def __init__(
        self,
        api_base_url: str,
        model_name: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        api_key: Optional[str] = None,
        timeout: int = 30,
        enable_kv_cache_prefill: bool = False,
        rag_template_str: Optional[str] = None,
        prefill_query: str = "Please understand this content",
        prefill_payload_builder: Optional[Callable[[str, str], dict]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_base_url = api_base_url.rstrip("/")
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.api_key = api_key
        self.timeout = timeout
        self.enable_kv_cache_prefill = enable_kv_cache_prefill
        self.rag_template_str = rag_template_str
        self.prefill_query = prefill_query
        self.prefill_payload_builder = prefill_payload_builder
        
        # Ensure chunk_overlap is less than chunk_size
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the tokenize endpoint.
        
        Supports multiple API types with fallback strategy:
        1. /v1/tokenize (preferred - faster, no LLM call)
        2. /v1/chat/completions (fallback)
        3. Character estimation (final fallback)
        
        Compatible with LlamaCPP and VLLM.
        """
        if not text or text.strip() == "":
            return 0

        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                # Strategy 1: Try /tokenize endpoint first (preferred)
                token_count = self._try_tokenize_endpoint(text, client, headers)
                if token_count is not None:
                    return token_count
                
                # Strategy 2: Fallback to /chat/completions
                log.debug("Tokenize endpoint not available, trying chat completions")
                token_count = self._try_chat_completions_endpoint(text, client, headers)
                if token_count is not None:
                    return token_count
                
                # Strategy 3: Final fallback to character estimation
                log.warning("Both tokenize and chat completions failed, using character estimation")
                return len(text) // 4
                    
        except httpx.TimeoutException:
            log.error(f"Timeout when counting tokens for text of length {len(text)}")
            return len(text) // 4  # Fallback estimation
        except Exception as e:
            log.error(f"Error counting tokens: {e}")
            return len(text) // 4  # Fallback estimation
    
    def _try_tokenize_endpoint(self, text: str, client: httpx.Client, headers: dict) -> Optional[int]:
        """
        Try to count tokens using the /tokenize endpoint.
        Returns None if the endpoint is not available.
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": text,
            }
            response = client.post(
                f"{self.api_base_url.split('/v1')[0]}/tokenize",
                json=payload,
                headers=headers,
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # VLLM format: {"tokens": [1, 2, 3, ...]}
                tokens = data.get("tokens", [])
                if isinstance(tokens, list) and len(tokens) > 0:
                    log.debug(f"Tokenized via /tokenize endpoint: {len(tokens)} tokens")
                    return len(tokens)
                
                # Alternative format: {"count": 123}
                count = data.get("count")
                if count is not None:
                    log.debug(f"Tokenized via /tokenize endpoint: {count} tokens")
                    return count
                    
            elif response.status_code == 404:
                log.debug("/tokenize endpoint not found")
                return None
            else:
                log.debug(f"/tokenize returned {response.status_code}")
                return None
                
        except Exception as e:
            log.debug(f"Tokenize endpoint failed: {e}")
            return None
    
    def _try_chat_completions_endpoint(self, text: str, client: httpx.Client, headers: dict) -> Optional[int]:
        """
        Try to count tokens using the /chat/completions endpoint.
        Returns None if the endpoint is not available.
        """
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": text}
                ],
                "max_tokens": 1,  # Minimal generation
                "temperature": 0,
                "stream": False,
            }
            
            response = client.post(
                f"{self.api_base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract token count from usage field (OpenAI-compatible format)
                usage = data.get("usage", {})
                
                # Try standard OpenAI format
                prompt_tokens = usage.get("prompt_tokens", 0)
                if prompt_tokens > 0:
                    log.debug(f"Tokenized via /chat/completions: {prompt_tokens} tokens")
                    return prompt_tokens
                
                # Try LlamaCPP specific fields
                prompt_eval_count = usage.get("prompt_eval_count", 0)
                if prompt_eval_count > 0:
                    log.debug(f"Tokenized via /chat/completions (LlamaCPP): {prompt_eval_count} tokens")
                    return prompt_eval_count
                
                log.warning(f"Could not find token count in chat completions response: {usage}")
                return None
                
            elif response.status_code == 404:
                log.debug("/chat/completions endpoint not found")
                return None
            else:
                log.debug(f"/chat/completions returned {response.status_code}")
                return None
                
        except Exception as e:
            log.debug(f"Chat completions endpoint failed: {e}")
            return None

    def _prefill_kv_cache(self, document_content: str) -> bool:
        """
        Prefill KV Cache by sending the document with RAG template to LLM.
        This allows the LLM to cache the context for faster subsequent queries.
        
        The payload structure follows EXACTLY the same pattern as process_chat_payload in middleware.py,
        using rag_template() and add_or_update_system_message() for compatibility.
        
        Reference: middleware.py lines 1176-1181
        
        Args:
            document_content: The document text to prefill
            
        Returns:
            True if prefill was successful, False otherwise
        """
        if not self.enable_kv_cache_prefill:
            return False
            
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            # Use custom payload builder if provided
            if self.prefill_payload_builder:
                try:
                    payload = self.prefill_payload_builder(self.model_name, document_content)
                    log.debug(f"Using custom payload builder for KV Cache prefill")
                except Exception as e:
                    log.error(f"Custom payload builder failed: {e}, falling back to default")
                    payload = None
            else:
                payload = None
            
            # Default payload construction if custom builder not provided or failed
            # This follows EXACTLY the same pattern as middleware.py process_chat_payload
            if payload is None:
                messages = []
                
                # Step 1: Format the document content using rag_template
                # This is EXACTLY what middleware.py does at lines 1177-1179
                if self.rag_template_str:
                    # Use rag_template to format context and query
                    # context_string = document_content (the RAG context)
                    # prompt = self.prefill_query (placeholder user query)
                    formatted_content = rag_template(
                        self.rag_template_str,  # template (like request.app.state.config.RAG_TEMPLATE)
                        document_content,        # context_string (the document content)
                        self.prefill_query      # prompt (placeholder query)
                    )
                    
                    # Step 2: Add to messages using add_or_update_system_message
                    # This is EXACTLY what middleware.py does at line 1176
                    messages = add_or_update_system_message(
                        formatted_content,
                        messages
                    )
                else:
                    # Fallback: Simple format without RAG template
                    messages.append({
                        "role": "system",
                        "content": f"Context:\n{document_content}\n\nPlease understand and remember this context."
                    })
                
                # Add a minimal user message to complete the conversation flow
                if not any(msg.get("role") == "user" for msg in messages):
                    messages.append({
                        "role": "user",
                        "content": self.prefill_query
                    })
                
                # Build payload following the same structure as in generate_chat_completion
                # Reference: middleware.py process_chat_payload and chat_completion flow
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 1,  # Minimal generation to save resources
                    "temperature": 0,  # Deterministic for caching
                    "stream": False,  # Non-streaming for prefill
                }
                
                # Optional: Add metadata for tracking (similar to middleware.py)
                # This helps identify prefill requests in logs
                payload["metadata"] = {
                    "task": "kv_cache_prefill",
                    "document_length": len(document_content),
                }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.api_base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                
                if response.status_code == 200:
                    log.debug(f"KV Cache prefilled for document (length: {len(document_content)})")
                    return True
                else:
                    log.warning(f"KV Cache prefill failed with status {response.status_code}: {response.text[:200]}")
                    return False
                    
        except httpx.TimeoutException:
            log.error(f"Timeout during KV Cache prefill for document of length {len(document_content)}")
            return False
        except Exception as e:
            log.error(f"Error during KV Cache prefill: {e}")
            return False


    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count.
        """
        if not text or text.strip() == "":
            return []

        # Split by paragraph first to avoid breaking mid-sentence
        separators = ["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
        
        return self._split_text_recursive(text, separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using different separators until chunk size is met.
        """
        if not separators:
            # Base case: no more separators, split by character count
            return self._split_by_tokens(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        # Merge small splits and split large ones
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for split in splits:
            if not split:
                continue
                
            split_with_sep = split + separator if separator else split
            split_tokens = self._count_tokens(split_with_sep)
            
            # If single split is too large, recursively split it
            if split_tokens > self.chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Recursively split the large piece
                sub_chunks = self._split_text_recursive(split, remaining_separators)
                chunks.extend(sub_chunks)
                continue
            
            # Try to add to current chunk
            if current_tokens + split_tokens <= self.chunk_size:
                current_chunk.append(split_with_sep)
                current_tokens += split_tokens
            else:
                # Current chunk is full, save it and start new one
                if current_chunk:
                    chunks.append("".join(current_chunk))
                
                # Start new chunk with overlap if needed
                if self.chunk_overlap > 0 and current_chunk:
                    # Calculate overlap from previous chunk
                    overlap_text = self._get_overlap_text(
                        "".join(current_chunk), 
                        self.chunk_overlap
                    )
                    current_chunk = [overlap_text, split_with_sep]
                    current_tokens = self._count_tokens(overlap_text) + split_tokens
                else:
                    current_chunk = [split_with_sep]
                    current_tokens = split_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks

    def _split_by_tokens(self, text: str) -> List[str]:
        """
        Split text into chunks of specified token size (character-level fallback).
        """
        # This is a character-level split when we run out of separators
        chunks = []
        current_pos = 0
        text_len = len(text)
        
        while current_pos < text_len:
            # Estimate character count for chunk_size tokens (1 token ≈ 4 chars)
            estimated_chars = self.chunk_size * 4
            end_pos = min(current_pos + estimated_chars, text_len)
            
            chunk = text[current_pos:end_pos]
            chunk_tokens = self._count_tokens(chunk)
            
            # Adjust if needed
            while chunk_tokens > self.chunk_size and end_pos > current_pos + 1:
                end_pos = int(end_pos * 0.9)  # Reduce by 10%
                chunk = text[current_pos:end_pos]
                chunk_tokens = self._count_tokens(chunk)
            
            chunks.append(chunk)
            
            # Calculate overlap for next chunk
            if self.chunk_overlap > 0:
                overlap_chars = self.chunk_overlap * 4  # Estimate
                current_pos = max(current_pos + 1, end_pos - overlap_chars)
            else:
                current_pos = end_pos
        
        return chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """
        Get the last N tokens worth of text for overlap.
        """
        if overlap_tokens <= 0:
            return ""
        
        # Estimate characters for overlap tokens
        estimated_chars = overlap_tokens * 4
        
        if len(text) <= estimated_chars:
            return text
        
        # Get approximate overlap text
        overlap_text = text[-estimated_chars:]
        
        # Verify and adjust
        actual_tokens = self._count_tokens(overlap_text)
        
        # Adjust if needed
        attempts = 0
        while actual_tokens > overlap_tokens and len(overlap_text) > 0 and attempts < 5:
            overlap_text = overlap_text[len(overlap_text) // 10:]  # Remove 10% from start
            actual_tokens = self._count_tokens(overlap_text)
            attempts += 1
        
        return overlap_text

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks, merging small documents when possible.
        
        Strategy:
        1. Try to merge small documents together to reach chunk_size
        2. Split documents that are too large
        3. Keep metadata from all merged documents
        """
        if not documents:
            return []
        
        result_docs = []
        current_text = ""
        current_metadatas = []  # List of metadata dicts from merged documents
        current_tokens = 0
        
        for doc in documents:
            doc_text = doc.page_content
            doc_tokens = self._count_tokens(doc_text)
            
            # If this single document is larger than chunk_size, split it independently
            if doc_tokens > self.chunk_size:
                # First, save any accumulated content
                if current_text:
                    result_docs.append(
                        Document(
                            page_content=current_text,
                            metadata=self._merge_metadatas(current_metadatas)
                        )
                    )
                    current_text = ""
                    current_metadatas = []
                    current_tokens = 0
                
                # Split the large document
                chunks = self.split_text(doc_text)
                for chunk in chunks:
                    result_docs.append(
                        Document(
                            page_content=chunk,
                            metadata=doc.metadata.copy()
                        )
                    )
                continue
            
            # Try to merge with current accumulated content
            if current_tokens + doc_tokens <= self.chunk_size:
                # Can merge
                separator = "\n\n" if current_text else ""
                current_text += separator + doc_text
                current_metadatas.append(doc.metadata.copy())
                current_tokens += doc_tokens + (self._count_tokens(separator) if separator else 0)
            else:
                # Cannot merge, save current and start new
                if current_text:
                    result_docs.append(
                        Document(
                            page_content=current_text,
                            metadata=self._merge_metadatas(current_metadatas)
                        )
                    )
                
                # Start new chunk with current document
                current_text = doc_text
                current_metadatas = [doc.metadata.copy()]
                current_tokens = doc_tokens
        
        # Don't forget the last accumulated content
        if current_text:
            result_docs.append(
                Document(
                    page_content=current_text,
                    metadata=self._merge_metadatas(current_metadatas)
                )
            )
        
        # Prefill KV Cache if enabled
        if self.enable_kv_cache_prefill and result_docs:
            log.info(f"Prefilling KV Cache for {len(result_docs)} document chunks...")
            success_count = 0
            for i, doc in enumerate(result_docs):
                if self._prefill_kv_cache(doc.page_content):
                    success_count += 1
                if (i + 1) % 10 == 0:  # Log progress every 10 documents
                    log.info(f"Prefilled {i + 1}/{len(result_docs)} documents...")
            log.info(f"KV Cache prefill completed: {success_count}/{len(result_docs)} successful")
        
        return result_docs
    
    def _merge_metadatas(self, metadatas: List[dict]) -> dict:
        """
        Merge multiple metadata dictionaries from merged documents.
        
        Strategy:
        - If only one metadata, return it as is
        - If multiple, create a combined metadata with lists of unique values
        - Special handling for common fields like 'source', 'name', 'file_id'
        """
        if not metadatas:
            return {}
        
        if len(metadatas) == 1:
            return metadatas[0]
        
        # Merge multiple metadatas
        merged = {}
        
        # Collect all keys
        all_keys = set()
        for m in metadatas:
            all_keys.update(m.keys())
        
        for key in all_keys:
            values = []
            for m in metadatas:
                if key in m:
                    value = m[key]
                    # Avoid duplicates
                    if value not in values:
                        values.append(value)
            
            # Store as single value if only one unique value, otherwise as list
            if len(values) == 1:
                merged[key] = values[0]
            else:
                merged[key] = values
        
        # Add a special flag to indicate this is a merged document
        merged['_merged'] = True
        merged['_source_count'] = len(metadatas)
        
        return merged

