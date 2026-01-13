import copy
import math
import os
import pprint
import importlib
from typing import Optional, Literal

import torch
import logging
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
from transformers.utils import add_start_docstrings

from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.token_grammar_recognizer import BaseTokenRecognizer

logger = logging.getLogger(__name__)


class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        grammar_constraint: BaseTokenRecognizer,
        valid_token_start_idx: Optional[int] = None,
        execution_mode: Literal["speculation", "full_mask"] = "full_mask",
        device: Optional[torch.device] = None,
        adapter: str = "transformers",
    ) -> None:
        self.last_size = None
        self.grammar_constraint = grammar_constraint
        self.batch_parsing_states = None
        self.valid_token_start_idx = valid_token_start_idx
        self.execution_mode = execution_mode
        self.device = device
        self._vocab_mismatch_logged = False  # Flag to log warning only once

        # Create an alias for llama-cpp-python
        if adapter == "llama-cpp-python":
            adapter = "llama_cpp_python"

        self.adapter = adapter

        # Load adapter if specified and not "transformers"
        self._adapter_func = None
        if adapter != "transformers":
            try:
                # Import the adapter module
                adapter_module = importlib.import_module(
                    f"transformers_cfg.adapters.{adapter}"
                )
                # Get the adapter function with the same name as the module
                adapter_func = getattr(adapter_module, adapter)
                # Create the adapter function with this processor
                self._adapter_func = adapter_func(self)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    f"Failed to load adapter '{adapter}': {str(e)}. "
                    f"Falling back to default transformers behavior."
                )

    def _compute_speculative_acceptance(
        self, logits: torch.FloatTensor, device: torch.device
    ) -> torch.Tensor:
        """
        Try accepting the most likely token first (speculation mode).

        If the most likely token is accepted by the grammar, only mark that token
        as accepted. Otherwise, fall back to computing full vocabulary acceptance.
        This can be faster when the grammar rarely rejects tokens.

        Args:
            logits: Unmasked logits tensor of shape (batch_size, vocab_size)
            device: Device to create tensors on

        Returns:
            Boolean acceptance tensor of shape (batch_size, vocab_size)
        """
        acceptance = torch.zeros(
            (logits.shape[0], len(self.grammar_constraint.token2byte_mapping)),
            dtype=torch.bool,
            device=device,
        )
        next_tokens = torch.argmax(logits, dim=-1)

        for i, next_token in enumerate(next_tokens.tolist()):
            try:
                is_next_token_accepted = self.grammar_constraint.accept_token_ids(
                    [next_token], self.batch_parsing_states[i]
                )
            except ValueError:
                is_next_token_accepted = False

            if is_next_token_accepted:
                acceptance[i, next_token] = True
            else:
                # Most likely token rejected, compute full acceptance mask
                acceptance[i] = self.grammar_constraint.filter_vocab(
                    self.batch_parsing_states[i], device
                )

        return acceptance

    def _compute_full_acceptance(self, device: torch.device) -> torch.Tensor:
        """
        Compute acceptance mask for all tokens (full mask mode).

        Tests every token in the vocabulary against the current grammar state
        to determine which tokens are acceptable. This is comprehensive but
        more computationally expensive than speculation mode.

        Args:
            device: Device to create tensors on

        Returns:
            Boolean acceptance tensor of shape (batch_size, vocab_size)
        """
        return self.grammar_constraint.batch_filter_vocab(
            self.batch_parsing_states, device
        )

    def _compute_acceptance_mask(
        self, logits: torch.FloatTensor, device: torch.device
    ) -> torch.Tensor:
        """
        Compute token acceptance mask using current execution mode.

        Args:
            logits: Unmasked logits tensor
            device: Device to create tensors on

        Returns:
            Boolean acceptance tensor indicating which tokens are valid
        """
        if self.execution_mode == "speculation":
            return self._compute_speculative_acceptance(logits, device)
        else:
            return self._compute_full_acceptance(device)

    def _align_vocab_sizes(
        self,
        acceptance: torch.Tensor,
        masked_logits: torch.FloatTensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Handle vocabulary size mismatch between tokenizer and model.

        Some models have different vocabulary sizes than their tokenizers
        (e.g., due to padding for efficient computation). This method aligns
        the acceptance mask to match the model's logits dimension by either
        truncating or padding.

        Args:
            acceptance: Token acceptance mask from grammar constraint
            masked_logits: Model logits (used to determine target size)
            device: Device to create tensors on

        Returns:
            Acceptance tensor resized to match masked_logits dimensions

        Raises:
            RuntimeError: If automatic alignment fails
        """
        acceptance_vocab_size = acceptance.shape[-1]
        masked_logits_vocab_size = masked_logits.shape[-1]

        if masked_logits_vocab_size == acceptance_vocab_size:
            return acceptance  # No mismatch, return unchanged

        vocab_diff = acceptance_vocab_size - masked_logits_vocab_size

        # Log details and warning only once
        if not self._vocab_mismatch_logged:
            logger.warning(
                f"Vocab size mismatch detected: Model logits size = {masked_logits_vocab_size}, "
                f"Tokenizer/Acceptance mask size = {acceptance_vocab_size} (Difference: {vocab_diff})"
            )

            # Identify and log extra tokens
            if vocab_diff > 0:  # acceptance mask is larger
                extra_token_ids = list(
                    range(masked_logits_vocab_size, acceptance_vocab_size)
                )
                try:
                    # Attempt to decode extra tokens using the tokenizer
                    extra_tokens_str = self.grammar_constraint.tokenizer.decode(
                        extra_token_ids
                    )
                    logger.warning(
                        f"Tokenizer/Acceptance mask seems to have {vocab_diff} extra token IDs "
                        f"(IDs {extra_token_ids[0]} to {extra_token_ids[-1]}) compared to the model's logits dimension. "
                        f"Decoded extra tokens (approximate): '{extra_tokens_str}'. "
                        f"Truncating acceptance mask."
                    )
                except Exception as e:
                    logger.warning(
                        f"Tokenizer/Acceptance mask seems to have {vocab_diff} extra token IDs "
                        f"(IDs {extra_token_ids[0]} to {extra_token_ids[-1]}) compared to the model's logits dimension, "
                        f"but decoding failed: {e}. Truncating acceptance mask."
                    )
            else:  # model logits is larger (vocab_diff < 0)
                extra_token_ids = list(
                    range(acceptance_vocab_size, masked_logits_vocab_size)
                )
                logger.warning(
                    f"Model logits dimension seems to be {abs(vocab_diff)} larger than the tokenizer's vocab size. "
                    f"The extra model token IDs are from {extra_token_ids[0]} to {extra_token_ids[-1]}. "
                    f"Padding acceptance mask with 'False'. "
                    f"(Cannot decode model-specific token IDs from here)."
                )

            self._vocab_mismatch_logged = True

        # Apply the fix (runs every time a mismatch occurs)
        if vocab_diff > 0:  # acceptance mask is larger
            # Truncate the acceptance mask to match the logits size
            acceptance = acceptance[..., :masked_logits_vocab_size]
            logger.debug(f"Truncated acceptance mask to shape: {acceptance.shape}")
        else:  # model logits is larger (vocab_diff < 0)
            # Pad the acceptance mask with False to match the logits size
            num_padding = abs(vocab_diff)
            padding_tensor = torch.zeros(
                (*acceptance.shape[:-1], num_padding),
                dtype=torch.bool,
                device=device,
            )
            acceptance = torch.cat((acceptance, padding_tensor), dim=-1)
            logger.debug(f"Padded acceptance mask to shape: {acceptance.shape}")

        # Final check to ensure shapes now match after correction
        if acceptance.shape[-1] != masked_logits.shape[-1]:
            raise RuntimeError(
                f"Automatic vocab size alignment failed. "
                f"Adjusted acceptance shape: {acceptance.shape}, "
                f"Masked logits shape: {masked_logits.shape}"
            )

        return acceptance

    def _log_accepted_tokens(self, acceptance: torch.Tensor) -> None:
        """
        Log debug information about accepted tokens.

        Only active when DEBUG_MODE environment variable is set to "True".
        Logs both token indices and decoded token strings for each batch element.

        Args:
            acceptance: Boolean acceptance tensor of shape (batch_size, vocab_size)
        """
        if os.getenv("DEBUG_MODE") != "True":
            return

        batch_size, vocab_size = acceptance.shape
        acceptance_np = acceptance.cpu().numpy()
        accepted_x, accepted_y = acceptance_np.nonzero()

        # Build dict of {batch_index: [accepted_token_indices]}
        accepted_token_indices = {i: [] for i in range(batch_size)}
        for x, y in zip(accepted_x, accepted_y):
            accepted_token_indices[x].append(y)

        logger.debug("Accepted token indices for the current batch:")
        logger.debug("\n" + pprint.pformat(accepted_token_indices))

        # Convert token IDs to token strings
        accepted_tokens = {
            i: [
                self.grammar_constraint.tokenizer.decode([token_id])
                for token_id in token_ids
            ]
            for i, token_ids in accepted_token_indices.items()
        }
        logger.debug("Accepted tokens for the current batch:")
        logger.debug("\n" + pprint.pformat(accepted_tokens))

    def mask_logits(
        self, logits: torch.FloatTensor, device: torch.device
    ) -> torch.FloatTensor:
        """
        Apply grammar constraints by masking invalid tokens in logits.

        This is the main entry point for logit masking. It computes which tokens
        are acceptable according to the grammar, handles any vocabulary size
        mismatches, and sets invalid token logits to negative infinity.

        Args:
            logits: Unmasked logits tensor of shape (batch_size, vocab_size)
            device: Device to perform computations on

        Returns:
            Masked logits tensor with invalid tokens set to -inf
        """
        masked_logits = logits.clone()

        # Compute which tokens are acceptable
        acceptance = self._compute_acceptance_mask(logits, device)

        # Handle vocabulary size mismatches
        acceptance = self._align_vocab_sizes(acceptance, masked_logits, device)

        # Log debug information if enabled
        self._log_accepted_tokens(acceptance)

        # Mask invalid tokens by setting their logits to negative infinity
        masked_logits[~acceptance] = -math.inf

        return masked_logits

    def process_logits(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        :param input_ids:
        :param scores:
        :return:
        """
        if self.device is None:
            device = scores.device
        else:
            device = self.device  # Use explicitly set device if available

        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_parsing_states is None:
            self.batch_parsing_states = [
                # self.grammar_constraint.init_stacks()
                copy.deepcopy(
                    self.grammar_constraint.string_recognizer.get_initial_parsing_state()
                )
                for _ in range(len(input_ids))
            ]

        if os.getenv("DEBUG_MODE") == "True":
            print("-" * 80)

        logger.debug("input_ids: \n" + pprint.pformat(input_ids))
        # logger.debug("scores: \n" + pprint.pformat(scores))
        logger.debug("last_size: \n" + pprint.pformat(self.last_size))
        logger.debug(
            "num of stacks: \n"
            + pprint.pformat(
                [len(acc_state.stacks) for acc_state in self.batch_parsing_states]
            )
        )
        # logger.debug("stacks: \n" + pprint.pformat(self.batch_parsing_states.stacks))

        self.batch_parsing_states = (
            self.grammar_constraint.update_state_with_batch_token_seqs(
                input_ids, self.batch_parsing_states, self.valid_token_start_idx
            )
        )
        logger.debug(f"input_ids: {input_ids}")

        masked_scores = self.mask_logits(scores, device)
        return masked_scores

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids, scores):
        # If we have an adapter function, use it
        if self._adapter_func is not None:
            return self._adapter_func(input_ids, scores)
        # Otherwise, use the default behavior
        return self.process_logits(input_ids, scores)

    def reset(self):
        self.batch_parsing_states = None
        self._vocab_mismatch_logged = False  # Reset flag on reset
        if isinstance(self.grammar_constraint, IncrementalGrammarConstraint):
            self.grammar_constraint.reset()
