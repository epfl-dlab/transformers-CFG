"""Centralized tokenizer family detection and classification."""

import logging
from typing import Literal

from transformers import (
    GPT2TokenizerFast,
    BartTokenizerFast,
    LlamaTokenizerFast,
    T5TokenizerFast,
    CodeGenTokenizerFast,
    PreTrainedTokenizerFast,
    GemmaTokenizerFast,
    Qwen2TokenizerFast,
    ByT5Tokenizer,
)

logger = logging.getLogger(__name__)

TokenizerFamily = Literal["gpt2", "llama1", "llama3", "t5", "byt5", "codegen", "unsupported"]


class TokenizerInspector:
    """
    Centralized tokenizer family detection and classification.

    This class consolidates all tokenizer type checking logic to eliminate
    duplication across the codebase. It handles special cases like deepseek-coder
    and Llama-3 family detection.

    Attributes:
        tokenizer: HuggingFace tokenizer to inspect

    Example:
        >>> inspector = TokenizerInspector(tokenizer)
        >>> family = inspector.get_family()
        >>> if inspector.is_deepseek_coder():
        ...     # Handle deepseek special case
    """

    def __init__(self, tokenizer):
        """
        Initialize inspector with a tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer instance to inspect
        """
        self.tokenizer = tokenizer

    def is_deepseek_coder(self) -> bool:
        """
        Check if tokenizer is from deepseek-coder model.

        DeepSeek-coder is a special case that inherits from LlamaTokenizerFast
        but actually behaves like GPT2TokenizerFast.

        Returns:
            True if tokenizer is deepseek-coder
        """
        return "deepseek-coder" in self.tokenizer.name_or_path

    def is_llama3_family(self) -> bool:
        """
        Check if tokenizer is from Llama-3 family.

        Includes llama-3, llama-3.1, llama-3.2, llama-3.3, etc.

        Returns:
            True if tokenizer is from Llama-3 family
        """
        return "Llama-3" in self.tokenizer.name_or_path

    def get_family(self) -> TokenizerFamily:
        """
        Detect tokenizer family.

        Returns:
            Tokenizer family string identifier

        Note:
            This consolidates all tokenizer detection logic that was previously
            duplicated across token2byte.py and tokenizer.py.
        """
        # Check GPT2-style tokenizers (including special cases)
        if isinstance(
            self.tokenizer,
            (GPT2TokenizerFast, BartTokenizerFast, Qwen2TokenizerFast),
        ):
            return "gpt2"

        # Check CodeGen tokenizers (used by Phi models)
        if isinstance(self.tokenizer, CodeGenTokenizerFast):
            return "codegen"

        # Check Llama family (with special cases)
        if isinstance(self.tokenizer, (LlamaTokenizerFast, GemmaTokenizerFast)):
            # DeepSeek-coder inherits from LlamaTokenizerFast but is actually GPT2-style
            if self.is_deepseek_coder():
                return "gpt2"
            return "llama1"

        # Check T5 tokenizers
        if isinstance(self.tokenizer, T5TokenizerFast):
            return "t5"

        # Check ByT5 tokenizers (character-level)
        if isinstance(self.tokenizer, ByT5Tokenizer):
            return "byt5"

        # Check Llama-3 family (uses PreTrainedTokenizerFast base)
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            if self.is_llama3_family():
                return "llama3"  # Llama-3 behaves like GPT2

        return "unsupported"

    def is_supported(self) -> bool:
        """
        Check if tokenizer is supported.

        Returns:
            True if tokenizer family is supported
        """
        return self.get_family() != "unsupported"

    def requires_bos_space_removal(self) -> bool:
        """
        Check if tokenizer requires leading space removal at BOS.

        Some tokenizers (LLAMA1, T5) add a leading space character that
        needs to be removed at the beginning of sentences.

        Returns:
            True if tokenizer requires BOS space removal
        """
        family = self.get_family()
        return family in ("llama1", "t5")

    def is_byte_level(self) -> bool:
        """
        Check if tokenizer operates at byte level.

        Returns:
            True if tokenizer is byte-level (GPT2-style, Llama-3, CodeGen)
        """
        family = self.get_family()
        return family in ("gpt2", "llama3", "codegen")

    def is_character_level(self) -> bool:
        """
        Check if tokenizer operates at character level.

        Returns:
            True if tokenizer is character-level (ByT5)
        """
        return self.get_family() == "byt5"
