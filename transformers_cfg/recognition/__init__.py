"""
Token recognition components for grammar-constrained generation.

This module provides components for incremental token recognition:

- BatchStateProcessor: Processes batch token sequences and updates parsing states
- TokenAcceptanceComputer: Computes which tokens are acceptable given parsing state

Usage:
    >>> from transformers_cfg.recognition import BatchStateProcessor, TokenAcceptanceComputer
    >>>
    >>> processor = BatchStateProcessor(recognizer, tokenizer, mapping, eos_id)
    >>> computer = TokenAcceptanceComputer(recognizer, trie, mapping, eos_id, use_unicode)
    >>>
    >>> states = processor.update_batch_states(input_ids, states)
    >>> acceptance = computer.compute_token_acceptance(state, device)
"""

from .batch_processor import BatchStateProcessor
from .acceptance_computer import TokenAcceptanceComputer

__all__ = ["BatchStateProcessor", "TokenAcceptanceComputer"]
