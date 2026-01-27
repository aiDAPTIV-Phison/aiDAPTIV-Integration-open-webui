from tokenizers import Tokenizer
from pathlib import Path
from typing import Dict, List, Optional

class TokenCounter:
    """
    Token counter using tokenizer library.

    This class wraps tokenizer functionality for counting tokens in text.
    Can be used directly or as a callable for length functions.
    Also provides methods for token-to-index conversion.
    """

    def __init__(self, tokenizer_path: str):
        """
        Initialize token counter with tokenizer file.

        Args:
            tokenizer_path: Path to tokenizer configuration file (JSON)

        Raises:
            FileNotFoundError: If tokenizer file does not exist

        Example:
            >>> counter = TokenCounter("tokenizer.json")
            >>> count = counter.count_tokens("Hello world")
            >>> # Or use as callable
            >>> length = counter("Hello world")
        """
        tokenizer_file = Path(tokenizer_path)
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer_path = tokenizer_path
        self._vocab: Optional[Dict[str, int]] = None
        self._id_to_token: Optional[Dict[int, str]] = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in given text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens

        Example:
            >>> counter = TokenCounter("tokenizer.json")
            >>> count = counter.count_tokens("Hello world")
            >>> print(count)  # e.g., 2
        """
        tokens = self.tokenizer.encode(str(text))
        return len(tokens.tokens)

    def __call__(self, text: str) -> int:
        """
        Make TokenCounter callable for use as length function.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens

        Example:
            >>> counter = TokenCounter("tokenizer.json")
            >>> length = counter("Hello world")  # Same as count_tokens()
        """
        return self.count_tokens(text)

    def get_vocab(self) -> Dict[str, int]:
        """
        獲取 token 到 index 的映射字典（詞彙表）。

        Returns:
            Dict[str, int]: token 到 index 的映射字典

        Example:
            >>> counter = TokenCounter("tokenizer.json")
            >>> vocab = counter.get_vocab()
            >>> print(vocab.get("hello", None))  # 獲取 "hello" 的 index
        """
        if self._vocab is None:
            self._vocab = self.tokenizer.get_vocab()
        return self._vocab

    def get_id_to_token(self) -> Dict[int, str]:
        """
        獲取 index 到 token 的映射字典（反向詞彙表）。

        Returns:
            Dict[int, str]: index 到 token 的映射字典

        Example:
            >>> counter = TokenCounter("tokenizer.json")
            >>> id_to_token = counter.get_id_to_token()
            >>> print(id_to_token.get(123, None))  # 獲取 index 123 對應的 token
        """
        if self._id_to_token is None:
            vocab = self.get_vocab()
            self._id_to_token = {v: k for k, v in vocab.items()}
        return self._id_to_token

    def token_to_index(self, token: str) -> Optional[int]:
        """
        將單個 token 轉換為對應的 index。

        Args:
            token: 要轉換的 token 字串

        Returns:
            Optional[int]: token 對應的 index，如果不存在則返回 None

        Example:
            >>> counter = TokenCounter("tokenizer.json")
            >>> index = counter.token_to_index("hello")
            >>> print(index)  # e.g., 123
        """
        vocab = self.get_vocab()
        return vocab.get(token, None)

    def index_to_token(self, index: int) -> Optional[str]:
        """
        將 index 轉換為對應的 token。

        Args:
            index: 要轉換的 token index

        Returns:
            Optional[str]: index 對應的 token，如果不存在則返回 None

        Example:
            >>> counter = TokenCounter("tokenizer.json")
            >>> token = counter.index_to_token(123)
            >>> print(token)  # e.g., "hello"
        """
        id_to_token = self.get_id_to_token()
        return id_to_token.get(index, None)

    def encode_to_ids(self, text: str) -> List[int]:
        """
        將文字編碼為 token IDs（index 列表）。

        Args:
            text: 要編碼的文字

        Returns:
            List[int]: token IDs 列表

        Example:
            >>> counter = TokenCounter("tokenizer.json")
            >>> ids = counter.encode_to_ids("Hello world")
            >>> print(ids)  # e.g., [123, 456]
        """
        encoding = self.tokenizer.encode(str(text))
        return encoding.ids

    def decode_from_ids(self, token_ids: List[int]) -> str:
        """
        將 token IDs（index 列表）解碼為文字。

        Args:
            token_ids: token IDs 列表

        Returns:
            str: 解碼後的文字

        Example:
            >>> counter = TokenCounter("tokenizer.json")
            >>> text = counter.decode_from_ids([123, 456])
            >>> print(text)  # e.g., "Hello world"
        """
        return self.tokenizer.decode(token_ids)

    def encode_to_tokens(self, text: str) -> List[str]:
        """
        將文字編碼為 token 字串列表。

        Args:
            text: 要編碼的文字

        Returns:
            List[str]: token 字串列表

        Example:
            >>> counter = TokenCounter("tokenizer.json")
            >>> tokens = counter.encode_to_tokens("Hello world")
            >>> print(tokens)  # e.g., ["Hello", " world"]
        """
        encoding = self.tokenizer.encode(str(text))
        return encoding.tokens
 
if __name__ == "__main__":
    token_counter = TokenCounter(tokenizer_path=r"D:\AI\Code\agentbuilder\km-for-agent-builder\Mistral-Small-24B-Instruct-2501_tokenizer.json")
    text = "Qwen3-4B-Thinking-2507\nChat\nHighlights\nOver the past three months, we have continued to scale the thinking capability of Qwen3-4B, improving both the quality and depth of reasoning. We are pleased to introduce Qwen3-4B-Thinking-2507, featuring the following key enhancements:\n\nSignificantly improved performance on reasoning tasks, including logical reasoning, mathematics, science, coding, and academic benchmarks that typically require human expertise.\nMarkedly better general capabilities, such as instruction following, tool usage,"
    print(token_counter.count_tokens(text))
    print(token_counter.encode_to_tokens(text))
    print(token_counter.encode_to_ids(text))
    print(token_counter.decode_from_ids(token_counter.encode_to_ids(text)))
    print(token_counter.token_to_index("hello"))
    print(token_counter.index_to_token(123))
    print(token_counter.get_vocab())
    print(token_counter.get_id_to_token())