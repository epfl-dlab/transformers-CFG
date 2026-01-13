"""
Parsing state management for batch grammar-constrained generation.

This module provides the ParsingStateManager class which handles initialization
and incremental updates of parsing states across batches during generation. It
ensures that each sequence in a batch maintains correct grammar state tracking
as new tokens are generated.

Key Components:
    ParsingStateManager: Orchestrates state initialization and updates for
        batch generation with grammar constraints

Architecture:
    The manager handles two phases of state management:
    1. Initialization: Create initial parsing states for each batch element,
       optionally skipping prefix tokens before valid_token_start_idx
    2. Updates: Incrementally update states as new tokens are generated,
       processing only new tokens to minimize computation

    State tracking uses the following strategy:
    - batch_parsing_states: List of AcceptState per batch element
    - last_size: Tracks input length to identify new tokens
    - Delegates token processing to grammar_constraint.update_state_with_batch_token_seqs()

Usage Patterns:
    1. Single generation run (typical):
       - Initialize states on first call with valid prefix
       - Incrementally update as each token is generated
       - Reset for next generation run

    2. Custom prefix handling:
       - Set valid_token_start_idx to skip prompt tokens
       - Grammar validation starts after this index
       - Useful for instruction-following models

Example:
    >>> from transformers_cfg.generation.parsing_state_manager import ParsingStateManager
    >>> from transformers_cfg.token_grammar_recognizer import IncrementalTokenRecognizer
    >>>
    >>> # Create state manager
    >>> manager = ParsingStateManager(
    ...     grammar_constraint=recognizer,
    ...     valid_token_start_idx=5  # Skip first 5 tokens
    ... )
    >>>
    >>> # Initialize and update states during generation
    >>> input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    >>> manager.update_states(input_ids)  # Initializes on first call
    >>> # States now track grammar compliance starting from token 6
    >>>
    >>> # Add more tokens
    >>> input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    >>> manager.update_states(input_ids)  # Incrementally processes token 8
    >>>
    >>> # Reset for next generation
    >>> manager.reset()

State Lifecycle:
    1. Manager created with grammar_constraint and optional valid_token_start_idx
    2. First update_states() call initializes batch_parsing_states
    3. Subsequent calls process only new tokens (incremental)
    4. reset() clears state for next generation run

Integration:
    Typically used by GrammarConstrainedLogitsProcessor, not directly by users.
    The processor calls update_states() before each logits masking operation.

Performance:
    - Incremental updates minimize redundant processing
    - Batch processing leverages GPU parallelization
    - Debug logging available with DEBUG_MODE=True environment variable

See Also:
    - logits_process.py: Uses ParsingStateManager for state coordination
    - ../recognition/batch_processor.py: Handles actual token sequence processing
    - ../recognizer.py: Defines AcceptState structure
"""

import copy
import logging
import os
import pprint
from typing import List, Optional

import torch

from transformers_cfg.recognizer import AcceptState
from transformers_cfg.token_grammar_recognizer import BaseTokenRecognizer

logger = logging.getLogger(__name__)


class ParsingStateManager:
    """
    Manages parsing states for batch grammar-constrained generation.

    This component handles initialization and updates of parsing states
    across a batch of sequences during generation. It coordinates with
    the token recognizer to maintain correct grammar state tracking.

    Attributes:
        grammar_constraint: Token recognizer for grammar checking
        valid_token_start_idx: Starting index for valid tokens
        batch_parsing_states: Current parsing states for each sequence
        last_size: Last processed input size (for incremental updates)

    Example:
        >>> manager = ParsingStateManager(grammar_constraint, valid_start_idx=0)
        >>> states = manager.update_states(input_ids)
    """

    def __init__(
        self,
        grammar_constraint: BaseTokenRecognizer,
        valid_token_start_idx: Optional[int] = None,
    ):
        """
        Initialize parsing state manager.

        Args:
            grammar_constraint: Token recognizer for grammar constraints
            valid_token_start_idx: Starting index for valid tokens in input
        """
        self.grammar_constraint = grammar_constraint
        self.valid_token_start_idx = valid_token_start_idx
        self.batch_parsing_states = None
        self.last_size = None

    def initialize_states(self, batch_size: int) -> List[AcceptState]:
        """
        Initialize parsing states for a new batch.

        Creates fresh parsing states for each sequence in the batch,
        each starting from the grammar's initial state.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            List of initial parsing states, one per sequence
        """
        self.batch_parsing_states = [
            copy.deepcopy(
                self.grammar_constraint.string_recognizer.get_initial_parsing_state()
            )
            for _ in range(batch_size)
        ]
        return self.batch_parsing_states

    def update_states(self, input_ids: torch.LongTensor) -> List[AcceptState]:
        """
        Update parsing states with new input tokens.

        Handles both initial batch processing and incremental updates.
        Initializes states on first call, then updates them with new tokens.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)

        Returns:
            Updated parsing states for each sequence

        Note:
            Debug logging is active when DEBUG_MODE environment variable is "True"
        """
        # Initialize states on first call
        if self.batch_parsing_states is None:
            self.initialize_states(len(input_ids))

        # Log debug information if enabled
        if os.getenv("DEBUG_MODE") == "True":
            print("-" * 80)
            logger.debug("input_ids: \n" + pprint.pformat(input_ids))
            logger.debug("last_size: \n" + pprint.pformat(self.last_size))
            logger.debug(
                "num of stacks: \n"
                + pprint.pformat(
                    [len(acc_state.stacks) for acc_state in self.batch_parsing_states]
                )
            )

        # Update parsing states with new tokens
        self.batch_parsing_states = (
            self.grammar_constraint.update_state_with_batch_token_seqs(
                input_ids, self.batch_parsing_states, self.valid_token_start_idx
            )
        )

        logger.debug(f"input_ids: {input_ids}")

        return self.batch_parsing_states

    def get_states(self) -> Optional[List[AcceptState]]:
        """
        Get current parsing states.

        Returns:
            Current batch parsing states, or None if not initialized
        """
        return self.batch_parsing_states

    def reset(self):
        """
        Reset manager state for processing a new batch.

        Clears all parsing states and resets internal tracking.
        """
        self.batch_parsing_states = None
        self.last_size = None
