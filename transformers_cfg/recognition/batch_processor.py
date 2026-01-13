"""Batch state processing for incremental token recognition."""

import logging
from typing import List, Optional

import torch
from transformers import PreTrainedTokenizer

from transformers_cfg.recognizer import AcceptState, StringRecognizer
from transformers_cfg.tokenization.mapping.token2byte import Token2ByteMapping

logger = logging.getLogger(__name__)


class BatchStateProcessor:
    """
    Processes batch token sequences and updates parsing states.

    This component handles the incremental processing of token sequences,
    maintaining parsing states for each sequence in a batch. It supports
    both single-token and multi-token updates.

    Attributes:
        string_recognizer: StringRecognizer for grammar-based parsing
        tokenizer: Tokenizer for decoding token sequences
        token2byte_mapping: Mapping from tokens to byte/codepoint sequences
        eos_token_id: End-of-sequence token ID
        last_size: Track last processed input size for incremental updates

    Example:
        >>> processor = BatchStateProcessor(
        ...     string_recognizer, tokenizer, token2byte_mapping, eos_token_id=2
        ... )
        >>> states = processor.update_batch_states(input_ids, states, valid_start_idx)
    """

    def __init__(
        self,
        string_recognizer: StringRecognizer,
        tokenizer: PreTrainedTokenizer,
        token2byte_mapping: Token2ByteMapping,
        eos_token_id: int,
    ):
        """
        Initialize batch state processor.

        Args:
            string_recognizer: StringRecognizer for parsing
            tokenizer: Tokenizer for decoding
            token2byte_mapping: Token to byte/codepoint mapping
            eos_token_id: ID of the end-of-sequence token
        """
        self.string_recognizer = string_recognizer
        self.tokenizer = tokenizer
        self.token2byte_mapping = token2byte_mapping
        self.eos_token_id = eos_token_id
        self.last_size = None

    def update_batch_states(
        self,
        input_ids: torch.LongTensor,
        batch_parsing_states: List[AcceptState],
        valid_token_start_idx: Optional[int] = None,
    ) -> List[AcceptState]:
        """
        Update parsing states for a batch of token sequences.

        Handles both initial processing (when last_size is None) and
        incremental processing (when adding one token at a time).

        Args:
            input_ids: Batch of token ID sequences
            batch_parsing_states: Current parsing states for each sequence
            valid_token_start_idx: Starting index for valid tokens (optional)

        Returns:
            Updated parsing states for each sequence in the batch

        Raises:
            RuntimeError: If input length is inconsistent with incremental processing
        """
        if self.last_size is None:
            # Initial processing: process all tokens from valid_token_start_idx
            valid_prefix_tokens = [
                (
                    single_input_ids[valid_token_start_idx:]
                    if valid_token_start_idx is not None
                    else []
                )
                for single_input_ids in input_ids
            ]

            batch_parsing_states = [
                self.update_single_sequence(prefix, parsing_state)
                for prefix, parsing_state in zip(
                    valid_prefix_tokens, batch_parsing_states
                )
            ]
        elif len(input_ids[0]) == self.last_size + 1:
            # Incremental processing: process only the last token
            batch_parsing_states = [
                self.update_with_token(single_input_ids[-1], parsing_state)
                for single_input_ids, parsing_state in zip(
                    input_ids, batch_parsing_states
                )
            ]
        else:
            # Input size is inconsistent - not processing one token at a time
            raise RuntimeError(
                "Input ID's length is inconsistent with the current state of "
                "the GrammarConstrainedLogitsProcessor. If you want to process "
                "another input sequence, please instantiate a new "
                "GrammarConstrainedLogitsProcessor."
            )

        self.last_size = len(input_ids[0])
        return batch_parsing_states

    def update_single_sequence(
        self,
        token_ids: List[int],
        parsing_state: Optional[AcceptState] = None,
        as_string: bool = True,
    ) -> AcceptState:
        """
        Update parsing state for a single token sequence.

        Args:
            token_ids: List of token IDs to process
            parsing_state: Current parsing state (None for initial state)
            as_string: Whether to decode as string (faster) or process token-by-token

        Returns:
            Updated parsing state after processing all tokens
        """
        if parsing_state is None:
            parsing_state = self.string_recognizer.get_initial_parsing_state()

        if as_string:
            # Fast path: decode entire sequence and process as string
            string = self.tokenizer.decode(token_ids)
            parsing_state = self.string_recognizer._update_state_with_string(
                string, parsing_state
            )
        else:
            # Slow path: process each token individually (for debugging)
            for i, token_id in enumerate(token_ids):
                parsing_state = self.update_with_token(token_id, parsing_state)
                if len(parsing_state.stacks) > 0:
                    cur_token_ids = token_ids[: i + 1]
                    logging.debug(f"{cur_token_ids} is accepted")
                    decoded_string = self.tokenizer.decode(cur_token_ids)
                    logging.debug(f"The decoded string is {decoded_string}")

        return parsing_state

    def update_with_token(
        self, token_id: int, parsing_state: AcceptState
    ) -> AcceptState:
        """
        Update parsing state with a single token.

        Args:
            token_id: Token ID to process
            parsing_state: Current parsing state

        Returns:
            Updated parsing state

        Raises:
            ValueError: If token is not accepted by current grammar state
        """
        if parsing_state.must_stop():
            if token_id == self.eos_token_id:
                return self.string_recognizer.get_termination_parsing_state()
            else:
                raise ValueError(
                    f"All stacks are empty, so the only token accepted is EOS({self.eos_token_id}), "
                    f"but got {token_id}. This error is likely due to the previous token not being "
                    f"accepted by the grammar."
                )

        if token_id == self.eos_token_id:
            if parsing_state.can_stop():
                # if at least one of the stack is empty, we can stop
                # we clear all the stacks, meaning that we don't accept any token after EOS
                return self.string_recognizer.get_termination_parsing_state()
            else:
                raise ValueError(
                    f"At least one of the stack should be empty when EOS is reached. "
                    f"However, the stacks are {parsing_state.stacks}"
                )

        bytes_or_codepoints = self.token2byte_mapping.map(token_id)
        parsing_state = self.string_recognizer._update_state_with_bytes(
            bytes_or_codepoints, parsing_state
        )
        return parsing_state

    def reset(self):
        """Reset the processor state for processing a new sequence."""
        self.last_size = None
