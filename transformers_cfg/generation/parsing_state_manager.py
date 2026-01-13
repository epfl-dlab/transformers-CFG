"""Parsing state management for batch grammar-constrained generation."""

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

    def update_states(
        self, input_ids: torch.LongTensor
    ) -> List[AcceptState]:
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
                    [
                        len(acc_state.stacks)
                        for acc_state in self.batch_parsing_states
                    ]
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
