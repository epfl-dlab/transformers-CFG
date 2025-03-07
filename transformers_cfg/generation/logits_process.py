import copy
import math
import os
import pprint
from typing import Optional, Literal

import numpy as np
import torch
import logging
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
from transformers.utils import add_start_docstrings

from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.token_grammar_recognizer import AbsTokenRecognizer

logger = logging.getLogger(__name__)


class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        grammar_constraint: AbsTokenRecognizer,
        valid_token_start_idx: Optional[int] = None,
        execution_mode: Literal["speculation", "full_mask"] = "full_mask",
        device: Optional[torch.device] = None,
        library: str = "transformers"
    ) -> None:
        self.last_size = None
        self.grammar_constraint = grammar_constraint
        self.batch_parsing_states = None
        self.valid_token_start_idx = valid_token_start_idx
        self.execution_mode = execution_mode
        self.device = device
        self.library = library
        if self.library == "llama-cpp-python":
            self.reinit_attempts = 0
            self.reinit_max = 3
            self.accumulated_tokens = []

    def mask_logits(
        self, logits: torch.FloatTensor, device: torch.device
    ) -> torch.FloatTensor:
        masked_logits = logits.clone()

        if self.execution_mode == "speculation":
            acceptance = torch.zeros(
                (logits.shape[0], len(self.grammar_constraint.homomorphism)),
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
                    acceptance[i] = self.grammar_constraint.filter_vocab(
                        self.batch_parsing_states[i], device
                    )
        else:
            acceptance = self.grammar_constraint.batch_filter_vocab(
                self.batch_parsing_states, device
            )

        acceptance_vocab_size = acceptance.shape[-1]
        masked_logits_vocab_size = masked_logits.shape[-1]
        if masked_logits_vocab_size != acceptance_vocab_size:
            assert (
                acceptance_vocab_size < masked_logits_vocab_size
            ), "impossible for tokenizer vocab to be less than model vocab"
            vocab_size_diff = masked_logits_vocab_size - acceptance_vocab_size
            false_tensor = torch.zeros(
                (*acceptance.shape[:-1], vocab_size_diff),
                dtype=torch.bool,
                device=device,
            )
            acceptance = torch.cat((acceptance, false_tensor), dim=-1)

        if os.getenv("DEBUG_MODE") == "True":
            batch_size, vocab_size = acceptance.shape
            acceptance_np = acceptance.cpu().numpy()
            accepted_x, accepted_y = acceptance_np.nonzero()
            accepted_token_indices = {i: [] for i in range(batch_size)}
            for x, y in zip(accepted_x, accepted_y):
                accepted_token_indices[x].append(y)
            logger.debug("Accepted token indices for the current batch:")
            logger.debug("\n" + pprint.pformat(accepted_token_indices))
            accepted_tokens = {
                i: [
                    self.grammar_constraint.tokenizer.decode([token_id])
                    for token_id in token_ids
                ]
                for i, token_ids in accepted_token_indices.items()
            }
            logger.debug("Accepted tokens for the current batch:")
            logger.debug("\n" + pprint.pformat(accepted_tokens))
        masked_logits[~acceptance] = -math.inf
        return masked_logits

    def process_logits(
        self, input_ids: list, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.device is None:
            device = scores.device
        if self.batch_parsing_states is None:
            self.batch_parsing_states = [
                copy.deepcopy(
                    self.grammar_constraint.string_recognizer.get_initial_parsing_state()
                )
                for _ in range(len(input_ids))
            ]
        logger.debug("input_ids: \n" + pprint.pformat(input_ids))
        logger.debug("last_size: \n" + pprint.pformat(self.last_size))
        logger.debug(
            "num of stacks: \n"
            + pprint.pformat(
                [len(acc_state.stacks) for acc_state in self.batch_parsing_states]
            )
        )
        self.batch_parsing_states = (
            self.grammar_constraint.update_state_with_batch_token_seqs(
                input_ids, self.batch_parsing_states, self.valid_token_start_idx
            )
        )
        logger.debug(f"input_ids: {input_ids}")
        masked_scores = self.mask_logits(scores, device)
        return masked_scores

    def _force_eos(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        eos_token = self.grammar_constraint.tokenizer.eos_token_id
        logger.warning(f"Forcing EOS token: {eos_token}")
        mask = torch.full_like(scores, fill_value=-float("inf"))
        if scores.dim() == 2:
            mask[:, eos_token] = 0
        else:
            mask[eos_token] = 0
        return mask

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids, scores
    ) -> torch.FloatTensor:
        if self.library == "llama-cpp-python":
            # Normalize input_ids to be a list of token sequences.
            if np.isscalar(input_ids):
                input_ids = [int(input_ids)]
            elif isinstance(input_ids, np.ndarray):
                input_ids = input_ids.tolist()
            elif isinstance(input_ids, list):
                input_ids = [int(i) if isinstance(i, np.generic) else i for i in input_ids]
            elif isinstance(input_ids, np.generic):
                input_ids = [int(input_ids)]
            if input_ids and isinstance(input_ids[0], int):
                input_ids = [input_ids]

            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores)
            if scores.dim() == 1:
                scores = scores.unsqueeze(0)

            # Track token accumulation for debugging.
            if len(input_ids[0]) > len(self.accumulated_tokens):
                new_token = input_ids[0][-1]
                self.accumulated_tokens.append(new_token)
                try:
                    token_text = self.grammar_constraint.tokenizer.decode([new_token])
                    logger.debug(f"Added token: {new_token} ({token_text})")
                except Exception:
                    logger.debug(f"Added token: {new_token} (cannot decode)")

            current_length = len(input_ids[0])
            if hasattr(self.grammar_constraint, "last_size") and self.grammar_constraint.last_size is not None:
                expected_length = self.grammar_constraint.last_size + 1
                if current_length != expected_length:
                    logger.warning(f"Length mismatch: current={current_length}, expected={expected_length}. Reinitializing grammar constraint.")
                    self.grammar_constraint.reset()
                    self.batch_parsing_states = None
                    self.reinit_attempts = 0
            try:
                processed_scores = self.process_logits(input_ids, scores)
                self.reinit_attempts = 0
            except ValueError as e:
                error_msg = str(e)
                if "All stacks are empty" in error_msg:
                    if self.reinit_attempts < self.reinit_max:
                        logger.warning(f"Grammar constraint error: {error_msg}. Attempt {self.reinit_attempts+1}/{self.reinit_max} to recover.")
                        self.grammar_constraint.reset()
                        self.batch_parsing_states = None
                        self.reinit_attempts += 1
                        try:
                            processed_scores = self.process_logits(input_ids, scores)
                        except ValueError as e2:
                            logger.error(f"Recovery failed: {str(e2)}")
                            processed_scores = self._force_eos(scores)
                    else:
                        logger.error(f"Max retries ({self.reinit_max}) exceeded. Forcing EOS.")
                        processed_scores = self._force_eos(scores)
                else:
                    logger.error(f"Unexpected error: {error_msg}")
                    raise e
            if processed_scores.dim() == 2 and processed_scores.size(0) == 1:
                processed_scores = processed_scores.squeeze(0)
            return processed_scores.detach().cpu().numpy()
        else:
            # Default transformers behavior.
            return self.process_logits(input_ids, scores)

    def reset(self):
        self.batch_parsing_states = None
        if isinstance(self.grammar_constraint, IncrementalGrammarConstraint):
            self.grammar_constraint.reset()
