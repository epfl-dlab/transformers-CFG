from abc import ABC, abstractmethod

import torch
from transformers_cfg.tokenization.SUPPORTED_TOKENIZERS import SUPPORTED_TOKENIZERS
from transformers_cfg.tokenization.tokenizer_inspector import TokenizerInspector
from .ByteProxyMapping import ByteProxyMapping, LLAMAByteProxyMapping
import logging
from transformers import (
    GPT2TokenizerFast,
    BartTokenizerFast,
    T5TokenizerFast,
    CodeGenTokenizerFast,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
    GemmaTokenizerFast,
    Qwen2TokenizerFast,
    ByT5Tokenizer,
)

from transformers_cfg.tokenization.utils import get_tokenizer_charset

log = logging.getLogger(__name__)


class Token2ByteMapping(ABC):
    """
    Base class for token-to-byte mapping with common template methods.

    This class provides template methods to reduce code duplication across
    different tokenizer families (GPT2, LLAMA1, T5, etc.).
    """

    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.tokenizer = tokenizer
        self.special = tokenizer.all_special_ids
        self._length = len(self.tokenizer.get_vocab())

    def __len__(self):
        return self._length

    @abstractmethod
    def map(self, token_id: int, verbose=False) -> bytes:
        pass

    def _normalize_token_id(self, token_id: int) -> int:
        """
        Convert token ID to integer if it's a tensor.

        Args:
            token_id: Token ID as int or tensor

        Returns:
            Token ID as integer
        """
        if isinstance(token_id, torch.Tensor):
            return token_id.item()
        return token_id

    def _is_special_token(self, token_id: int) -> bool:
        """
        Check if token ID is a special token.

        Args:
            token_id: Token ID to check

        Returns:
            True if token is special (BOS, EOS, PAD, etc.)
        """
        return token_id in self.special

    def _handle_special_token(self) -> bytes:
        """
        Handle special tokens (BOS, EOS, PAD, etc.).

        Returns:
            Empty bytes for special tokens
        """
        return b""

    def _get_proxy_token(self, token_id: int) -> str:
        """
        Get proxy token string for a token ID.

        Args:
            token_id: Token ID to convert

        Returns:
            Proxy token string representation
        """
        return self.tokenizer.convert_ids_to_tokens(token_id)

    @classmethod
    def from_hf_tokenizer(cls, hf_tokenizer):
        """
        Create appropriate mapping class for a HuggingFace tokenizer.

        Uses TokenizerInspector to centralize tokenizer family detection.

        Args:
            hf_tokenizer: HuggingFace tokenizer instance

        Returns:
            Appropriate Token2ByteMapping subclass instance

        Raises:
            AssertionError: If tokenizer type is not in SUPPORTED_TOKENIZERS
            NotImplementedError: If tokenizer family is not supported
        """
        assert (
            type(hf_tokenizer) in SUPPORTED_TOKENIZERS
        ), f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}, supported tokenizers: {SUPPORTED_TOKENIZERS}"

        # Use TokenizerInspector for centralized family detection
        inspector = TokenizerInspector(hf_tokenizer)
        family = inspector.get_family()

        if family == "gpt2" or family == "llama3":
            # GPT2-style tokenizers (includes Llama-3 and deepseek-coder)
            return GPT2Token2ByteMapping(hf_tokenizer)
        elif family == "llama1":
            return LLAMA1Token2ByteMapping(hf_tokenizer)
        elif family == "t5":
            return T5Token2ByteMapping(hf_tokenizer)
        elif family == "byt5":
            return ByT5Token2ByteMapping(hf_tokenizer)
        elif family == "codegen":
            # CodeGen uses GPT2-style mapping
            return GPT2Token2ByteMapping(hf_tokenizer)
        else:
            raise NotImplementedError(
                f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}"
            )

    @staticmethod
    def auto_infer(hf_tokenizer):
        "beta version, not sure if it will work for all cases"
        charset = get_tokenizer_charset(hf_tokenizer)
        size = len(charset)
        if size >= 256 and size < 256 + 30:
            return GPT2Token2ByteMapping(hf_tokenizer)
        elif "▁" in charset:
            return LLAMA1Token2ByteMapping(hf_tokenizer)
        else:
            raise NotImplementedError(
                f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}"
            )


class GPT2Token2ByteMapping(Token2ByteMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.byte_proxy_mapping = ByteProxyMapping(tokenizer)

    def map2proxy_token(self, token_id: int) -> str:
        """Get proxy token, handling special tokens."""
        # This is the case for BOS, EOS, etc.
        if self._is_special_token(token_id):
            return ""
        # Normalize token_id (convert tensor to int if needed)
        token_id = self._normalize_token_id(token_id)
        # Get proxy token representation
        return self._get_proxy_token(token_id)

    def map(self, token_id: int, verbose=False) -> bytes:
        proxy_token = self.map2proxy_token(token_id)
        if verbose:
            log.debug(f"token_id: {token_id}, token: {proxy_token}")

        return self.byte_proxy_mapping.map(proxy_token)


class LLAMA1Token2ByteMapping(Token2ByteMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.last_token_id = None
        self.byte_proxy_mapping = LLAMAByteProxyMapping()

    def map(self, token_id: int, verbose=False) -> bytes:
        # Check if we're at the beginning of the sentence
        # (need to remove space for BPE tokenizers)
        at_bos = False
        if self.last_token_id is not None and self.last_token_id == self.bos_token_id:
            at_bos = True
        self.last_token_id = token_id

        # Handle special tokens (BOS, EOS, PAD, etc.)
        if self._is_special_token(token_id):
            return self._handle_special_token()

        # Normalize token_id (convert tensor to int if needed)
        token_id = self._normalize_token_id(token_id)

        # Get proxy token and convert to bytes
        proxy_token = self._get_proxy_token(token_id)
        token_bytes = self.byte_proxy_mapping.map(proxy_token)

        # Remove leading space at beginning of sentence (BPE-specific)
        if token_bytes and token_bytes[0] == 32 and at_bos:
            token_bytes = token_bytes[1:]

        return token_bytes


class T5Token2ByteMapping(Token2ByteMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.at_bos = True
        self.byte_proxy_mapper = LLAMAByteProxyMapping()

    def map(self, token_id: int, verbose=False) -> bytes:
        # Handle special tokens (BOS, EOS, PAD, etc.)
        if self._is_special_token(token_id):
            self.at_bos = False
            return self._handle_special_token()

        # Normalize token_id (convert tensor to int if needed)
        token_id = self._normalize_token_id(token_id)

        # Get proxy token and convert to bytes
        proxy_token = self._get_proxy_token(token_id)
        token_bytes = self.byte_proxy_mapper.map(proxy_token)

        # Remove leading space at beginning of sentence (BPE-specific)
        if token_bytes and token_bytes[0] == 32 and self.at_bos:
            token_bytes = token_bytes[1:]

        self.at_bos = False
        return token_bytes


class ByT5Token2ByteMapping(Token2ByteMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.token_id_to_bytes = {}

    def map(self, token_id: int, verbose=False) -> bytes:
        # Normalize token_id (convert tensor to int if needed)
        token_id = self._normalize_token_id(token_id)

        # Check cache first
        if token_id in self.token_id_to_bytes:
            return self.token_id_to_bytes[token_id]

        # By inspecting the token vocab: first 3 tokens are special,
        # tokens after 258 are also special
        # Only tokens between 3 and 258 are valid (256 bytes)
        if 3 <= token_id <= 258:
            return ord(self.tokenizer.convert_ids_to_tokens(token_id)).to_bytes(
                1, "big"
            )
        else:
            # Return empty bytes for special tokens
            return bytes()
