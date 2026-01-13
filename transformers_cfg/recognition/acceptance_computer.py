"""
Token acceptance computation for grammar-constrained generation.

This module provides the TokenAcceptanceComputer class which determines which
tokens in the vocabulary are acceptable given the current parsing state. It uses
byte tries and string recognizers to efficiently compute acceptance masks for
constraining model generation.

Key Components:
    TokenAcceptanceComputer: Computes boolean acceptance tensors indicating
        which tokens are valid according to grammar constraints

Architecture:
    The computer operates in two modes:
    1. Unicode Mode (use_unicode=True):
       - Uses byte-by-byte validation via _can_accept_bytes()
       - Handles multibyte UTF-8 characters correctly
       - Works with byte_trie.get_next_token_acceptance()

    2. Non-Unicode Mode (use_unicode=False):
       - Uses trie-based token validation via check_token_acceptance_in_trie()
       - More efficient for ASCII-only grammars
       - Traverses byte trie while checking grammar acceptance

    Acceptance computation flow:
    1. For each stack in parsing_state.stacks:
       - Compute per-stack acceptance mask
       - Check each token's byte sequence against grammar
       - Handle EOS token specially (accepted if stack is empty)
    2. Merge stack results: token accepted if ANY stack accepts it
    3. Return boolean tensor of shape (vocab_size,)

Caching Strategy:
    Uses @lru_cache(maxsize=32768) on compute_for_single_stack() to avoid
    redundant computation for identical (stack, partial_utf8) pairs.
    Note: GPU cache memory persists even after object dereferencing.

Example:
    >>> from transformers_cfg.recognition.acceptance_computer import TokenAcceptanceComputer
    >>> from transformers_cfg.recognizer import StringRecognizer
    >>>
    >>> # Create computer
    >>> computer = TokenAcceptanceComputer(
    ...     string_recognizer=recognizer,
    ...     byte_trie=trie,
    ...     token2byte_mapping=mapping,
    ...     eos_token_id=2,
    ...     use_unicode=True
    ... )
    >>>
    >>> # Compute acceptance for current parsing state
    >>> state = recognizer.get_initial_parsing_state()
    >>> acceptance = computer.compute_token_acceptance(state, torch.device("cpu"))
    >>>
    >>> # Find acceptable tokens
    >>> valid_token_ids = acceptance.nonzero().squeeze().tolist()
    >>> print(f"Number of valid tokens: {len(valid_token_ids)}")

Stack Merging:
    When multiple parsing stacks exist (representing alternative parse paths),
    the computer computes acceptance for each stack independently, then merges:
    - acceptance_matrix shape: (num_stacks, vocab_size)
    - final acceptance: acceptance_matrix.any(dim=0)
    - Result: token accepted if valid on ANY parse path

EOS Handling:
    The EOS (end-of-sequence) token receives special treatment:
    - Accepted when parsing stack is empty (grammar complete)
    - Rejected when parsing stack is non-empty (grammar incomplete)
    - Prevents premature sequence termination

Integration:
    Used by IncrementalTokenRecognizer to provide token filtering capability.
    Typically invoked via get_next_token_acceptance() for backward compatibility.

Performance:
    - LRU cache dramatically reduces redundant computation
    - Cache size of 32768 balances memory vs. hit rate
    - GPU memory accumulation can occur with CUDA tensors
    - Consider clearing caches for long-running processes

See Also:
    - batch_processor.py: Updates parsing states processed by this computer
    - ../recognizer.py: Provides _can_accept_bytes() for unicode mode
    - ../token_grammar_recognizer.py: Uses this for check_token_acceptance_in_trie()
"""

import logging
from functools import lru_cache
from typing import Tuple

import torch

from transformers_cfg.recognizer import AcceptState, StringRecognizer
from transformers_cfg.tokenization.byte_trie import ByteTrie
from transformers_cfg.tokenization.mapping.token2byte import Token2ByteMapping
from transformers_cfg.utf8_utils import PartialUTF8

logger = logging.getLogger(__name__)


class TokenAcceptanceComputer:
    """
    Computes which tokens are acceptable given the current parsing state.

    This component is responsible for determining token acceptance by checking
    which tokens can be consumed according to the grammar rules. It uses a
    byte trie and string recognizer to efficiently compute acceptance masks.

    Attributes:
        string_recognizer: StringRecognizer for grammar state checking
        byte_trie: Trie structure mapping tokens to byte sequences
        token2byte_mapping: Mapping from tokens to byte/codepoint sequences
        eos_token_id: End-of-sequence token ID
        use_unicode: Whether the grammar uses unicode characters

    Example:
        >>> computer = TokenAcceptanceComputer(
        ...     string_recognizer, byte_trie, token2byte_mapping,
        ...     eos_token_id=2, use_unicode=True
        ... )
        >>> acceptance = computer.compute_token_acceptance(parsing_state, device)
    """

    def __init__(
        self,
        string_recognizer: StringRecognizer,
        byte_trie: ByteTrie,
        token2byte_mapping: Token2ByteMapping,
        eos_token_id: int,
        use_unicode: bool,
    ):
        """
        Initialize token acceptance computer.

        Args:
            string_recognizer: StringRecognizer for grammar checking
            byte_trie: Trie for token byte sequences
            token2byte_mapping: Token to byte/codepoint mapping
            eos_token_id: ID of the end-of-sequence token
            use_unicode: Whether grammar uses unicode
        """
        self.string_recognizer = string_recognizer
        self.byte_trie = byte_trie
        self.token2byte_mapping = token2byte_mapping
        self.eos_token_id = eos_token_id
        self.use_unicode = use_unicode

    def compute_token_acceptance(
        self, parsing_state: AcceptState, device: torch.device
    ) -> torch.Tensor:
        """
        Compute which tokens are acceptable for the current parsing state.

        Args:
            parsing_state: Current parsing state with stacks
            device: Device for tensor allocation

        Returns:
            Boolean tensor indicating which tokens are acceptable
        """
        acceptance_matrix = torch.stack(
            [
                self.compute_for_single_stack(
                    tuple(stack), parsing_state.partial_utf8, device
                )
                for stack in parsing_state.stacks
            ]
        )
        # Merge stacks: any True => True
        acceptance = acceptance_matrix.any(dim=0)
        return acceptance

    # If running on a GPU device this cache can continue to fill up GPU memory
    # Dereferencing the object will not clear the cache
    @lru_cache(maxsize=32768)
    def compute_for_single_stack(
        self, stack: Tuple[int], partial_utf8: PartialUTF8, device: torch.device
    ) -> torch.Tensor:
        """
        Compute token acceptance for a single parsing stack.

        This method is cached for performance. Each unique (stack, partial_utf8)
        combination is computed at most once.

        Args:
            stack: Single parsing stack (tuple of grammar offsets)
            partial_utf8: Partial UTF-8 sequence state
            device: Device for tensor allocation

        Returns:
            Boolean tensor indicating acceptable tokens for this stack
        """
        assert isinstance(stack, tuple)

        if self.use_unicode:
            accept_f = lambda x: self.string_recognizer._can_accept_bytes(
                x, {stack}, partial_utf8=partial_utf8
            )
            token_acceptance = self.byte_trie.get_next_token_acceptance(
                accept=accept_f, accept_eos=False, eos_token_id=self.eos_token_id
            )
        else:
            from transformers_cfg.token_grammar_recognizer import (
                check_token_acceptance_in_trie,
            )

            accepts = [False] * len(self.token2byte_mapping)
            token_acceptance = check_token_acceptance_in_trie(
                self.byte_trie.root,
                [stack],
                self.string_recognizer,
                self.eos_token_id,
                accepts,
            )

        x = torch.tensor(token_acceptance, dtype=torch.bool, device=device)
        x_eos = self._validate_and_set_eos_acceptance(x, stack)
        return x_eos

    def _validate_and_set_eos_acceptance(
        self, acceptance: torch.Tensor, stack: Tuple[int]
    ) -> torch.Tensor:
        """
        Set EOS token acceptance based on stack state.

        Args:
            acceptance: Current acceptance tensor
            stack: Parsing stack

        Returns:
            Updated acceptance tensor with EOS correctly set
        """
        if len(stack) == 0:
            # if the stack is empty, we can accept EOS
            acceptance[self.eos_token_id] = True
        return acceptance
