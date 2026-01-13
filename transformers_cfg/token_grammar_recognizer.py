import logging
from abc import ABC
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer

from transformers_cfg.recognizer import StringRecognizer, AcceptState
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.tokenization.byte_trie import ByteTrie, TrieNode
from transformers_cfg.tokenization.mapping.token2byte import (
    Token2ByteMapping,
)
from transformers_cfg.utf8_utils import PartialUTF8
from transformers_cfg.recognition import BatchStateProcessor, TokenAcceptanceComputer

logger = logging.getLogger(__name__)


class BaseTokenRecognizer(ABC):
    def __init__(
        self,
        grammar_str: str,
        tokenizer: PreTrainedTokenizer,
        start_rule_name: Optional[str] = "root",
        trie: Optional[ByteTrie] = None,
        token2byte_mapping: Optional[Token2ByteMapping] = None,
    ):
        parsed_grammar = parse_ebnf(grammar_str)
        grammar_encoding = parsed_grammar.grammar_encoding
        self.start_rule_id = parsed_grammar.symbol_table.get(start_rule_name)
        self.use_unicode = self.detect_unicode(grammar_str)

        self.eos_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.string_recognizer = StringRecognizer(grammar_encoding, self.start_rule_id)
        if trie is None:
            self.byte_trie = ByteTrie.from_tokenizer(tokenizer)
        else:
            self.byte_trie = trie
        if token2byte_mapping is None:
            self.token2byte_mapping = Token2ByteMapping.from_hf_tokenizer(tokenizer)
        else:
            self.token2byte_mapping = token2byte_mapping

    def try_accept_token_id(self, token_id: int, parsing_state: AcceptState) -> bool:
        if parsing_state.must_stop():
            if token_id == self.eos_token_id:
                return True
            else:
                return False
        if token_id == self.eos_token_id:
            if parsing_state.can_stop():
                # if at least one of the stack is empty, we can stop
                # we clear all the stacks, meaning that we don't accept any token after EOS
                return True
            else:
                return False
        # for code_point in self.mapping.map(token_id):
        #     stacks = self.grammar._consume_char_code_point(code_point, stacks)
        bytes_or_codepoints = self.token2byte_mapping.map(token_id, verbose=False)
        new_acc_state = self.string_recognizer._update_state_with_bytes(
            bytes_or_codepoints, parsing_state, verbose=False
        )
        return len(new_acc_state.stacks) > 0

    def update_state_with_batch_token_seqs(self, *args, **kwargs):
        """Process a list of tokens according to the grammar rules."""
        raise NotImplementedError

    def batch_filter_vocab(
        self, batch_parsing_states: List[AcceptState], device: torch.device
    ) -> torch.Tensor:
        batch_acceptance = []
        for parsing_state in batch_parsing_states:
            batch_acceptance.append(self.filter_vocab(parsing_state, device))
        return torch.stack(batch_acceptance)

    def filter_vocab(
        self, parsing_state: AcceptState, device: torch.device
    ) -> torch.Tensor:
        if not parsing_state.stacks:  # Check if stacks is empty
            # Handle the empty case: for example, return a tensor of False
            # The size of the tensor should match the size of your vocabulary
            vocab_size = len(self.token2byte_mapping)
            logger.debug(f"Empty stack, sum of acceptance: {0}")
            # size of the vocab
            accepts = [False] * vocab_size
            accepts[self.eos_token_id] = True
            return torch.tensor(accepts, dtype=torch.bool, device=device)

        acceptance = self.get_next_token_acceptance(parsing_state, device)
        return acceptance

    def get_next_token_acceptance(
        self, parsing_state: AcceptState, device: torch.device
    ) -> torch.Tensor:
        raise NotImplementedError

    def validate_and_set_eos_acceptance(
        self, acceptance: torch.Tensor, stack: Tuple[int]
    ) -> torch.Tensor:
        if len(stack) == 0:
            # if the stack is empty, we can accept EOS
            acceptance[self.eos_token_id] = True
        return acceptance

    def accept_token_ids(
        self,
        token_ids: List[int],
        parsing_state: Optional[AcceptState] = None,
        as_string: bool = True,
    ) -> bool:
        """Accept a list of token IDs according to the grammar rules."""
        raise NotImplementedError

    @staticmethod
    def detect_unicode(text: str) -> bool:
        # check if the text contains any unicode characters
        return any(ord(char) > 127 for char in text)


class IncrementalTokenRecognizer(BaseTokenRecognizer):
    """
    Incremental token recognizer using composed components.

    This class orchestrates BatchStateProcessor and TokenAcceptanceComputer
    to provide incremental token recognition with grammar constraints.

    Attributes:
        batch_processor: Handles batch state updates
        acceptance_computer: Computes token acceptance masks
    """

    def __init__(
        self,
        grammar_str: str,
        start_rule_name: str,
        tokenizer: PreTrainedTokenizer,
        trie: Optional[ByteTrie] = None,
        homomorphism: Optional[Token2ByteMapping] = None,
    ):
        super().__init__(
            grammar_str,
            tokenizer,
            start_rule_name,
            trie=trie,
            token2byte_mapping=homomorphism,
        )

        # Create composed components
        self._batch_processor = BatchStateProcessor(
            self.string_recognizer,
            self.tokenizer,
            self.token2byte_mapping,
            self.eos_token_id,
        )
        self._acceptance_computer = TokenAcceptanceComputer(
            self.string_recognizer,
            self.byte_trie,
            self.token2byte_mapping,
            self.eos_token_id,
            self.use_unicode,
        )

    # Properties for accessing composed components
    @property
    def batch_processor(self) -> BatchStateProcessor:
        """Access to batch processor component."""
        return self._batch_processor

    @property
    def acceptance_computer(self) -> TokenAcceptanceComputer:
        """Access to acceptance computer component."""
        return self._acceptance_computer

    @property
    def last_size(self):
        """Get last processed size (backward compatibility)."""
        return self._batch_processor.last_size

    @last_size.setter
    def last_size(self, value):
        """Set last processed size (backward compatibility)."""
        self._batch_processor.last_size = value

    def _update_state_with_token_id(
        self, token_id: int, parsing_state: AcceptState
    ) -> AcceptState:
        """
        Update state with single token (backward compatibility wrapper).

        Deprecated: New code should use batch_processor.update_with_token() directly.
        """
        return self._batch_processor.update_with_token(token_id, parsing_state)

    def update_state_with_batch_token_seqs(
        self,
        input_ids: torch.LongTensor,
        batch_parsing_states: list[AcceptState],
        valid_token_start_idx: Optional[int] = None,
    ) -> list[AcceptState]:
        """
        Update parsing states for batch of token sequences (backward compatibility wrapper).

        Deprecated: New code should use batch_processor.update_batch_states() directly.
        """
        return self._batch_processor.update_batch_states(
            input_ids, batch_parsing_states, valid_token_start_idx
        )

    def _update_state_with_single_token_seq(
        self,
        token_ids: List[int],
        parsing_state: Optional[AcceptState] = None,
        as_string=True,
    ) -> AcceptState:
        """
        Update state with token sequence (backward compatibility wrapper).

        Deprecated: New code should use batch_processor.update_single_sequence() directly.
        """
        return self._batch_processor.update_single_sequence(
            token_ids, parsing_state, as_string
        )

    def accept_token_ids(
        self,
        token_ids: List[int],
        parsing_state: Optional[AcceptState] = None,
        as_string: bool = True,
    ) -> bool:
        output_state = self._update_state_with_single_token_seq(
            token_ids, parsing_state, as_string
        )
        return len(output_state.stacks) > 0

    def get_next_token_acceptance(
        self, parsing_state: AcceptState, device: torch.device
    ) -> torch.Tensor:
        """
        Compute token acceptance (backward compatibility wrapper).

        Deprecated: New code should use acceptance_computer.compute_token_acceptance() directly.
        """
        return self._acceptance_computer.compute_token_acceptance(
            parsing_state, device
        )

    # If running on a GPU device this cache can continue to fill up GPU memory
    # Dereferencing the object will not clear the cache
    @lru_cache(maxsize=32768)
    def get_next_token_acceptance_for_single_stack(
        self, stack: Tuple[int], partial_utf8: PartialUTF8, device: torch.device
    ) -> torch.Tensor:
        """
        Compute token acceptance for single stack (backward compatibility wrapper).

        Deprecated: New code should use acceptance_computer.compute_for_single_stack() directly.
        """
        return self._acceptance_computer.compute_for_single_stack(
            stack, partial_utf8, device
        )

    def reset(self):
        """Reset processor state (backward compatibility wrapper)."""
        self._batch_processor.reset()


# def check_token_acceptance_in_trie(trie, stacks, grammar, eos_token_id, accepts):

#     for byte, next_trie in trie.items():
#         if byte == LEAF:
#             token_id = next_trie
#             if token_id != eos_token_id:
#                 # if the stacks is not empty, it means we can still continue to parse
#                 # so we should accept the token
#                 accepts[token_id] = bool(stacks)
#             continue

#         new_stacks = set()
#         for stk in stacks:
#             if not stk:
#                 continue

#             next_element_offset = stk[-1]
#             num_chars = grammar.grammar_encoding[next_element_offset]

#             if not grammar.char_acceptance_at_element(next_element_offset).get(
#                 byte, False
#             ):
#                 # if the current byte is not accepted by the current rule, we need to try next rule
#                 continue

#             next_element_offset += num_chars + 1
#             new_stack = list(stk[:-1])
#             if grammar.grammar_encoding[next_element_offset]:
#                 new_stack.append(next_element_offset)
#             new_stacks.update(grammar.expand_stack_head(tuple(new_stack)))

#         if new_stacks:
#             check_token_acceptance_in_trie(
#                 next_trie, new_stacks, grammar, eos_token_id, accepts
#             )

#     return accepts


def check_token_acceptance_in_trie(
    trie_node: TrieNode,
    stacks: List[Tuple[int]],
    recognizer: StringRecognizer,
    eos_token_id: int,
    accepts: List[bool],
) -> List[bool]:
    if trie_node.is_end_of_word:
        token_id = trie_node.token_id
        if token_id != eos_token_id:
            # if the stacks is not empty, it means we can still continue to parse
            # so we should accept the token
            accepts[token_id] = bool(stacks)

    for byte, next_trie_node in trie_node.children.items():
        new_stacks = set()
        for stk in stacks:
            if not stk:
                continue

            next_element_offset = stk[-1]
            num_chars = recognizer.grammar_encoding[next_element_offset]

            if not recognizer.char_acceptance_at_element(next_element_offset).get(
                byte, False
            ):
                # if the current byte is not accepted by the current rule, we need to try next rule
                continue

            next_element_offset += num_chars + 1
            new_stack = list(stk[:-1])
            if recognizer.grammar_encoding[next_element_offset]:
                new_stack.append(next_element_offset)
            new_stacks.update(recognizer.expand_stack_head(tuple(new_stack)))

        if new_stacks:
            check_token_acceptance_in_trie(
                next_trie_node, new_stacks, recognizer, eos_token_id, accepts
            )

    return accepts


class NonIncrementalTokenSeqRecognizer(IncrementalTokenRecognizer):
    def __init__(self, grammar_str, start_rule_name, tokenizer):
        super().__init__(grammar_str, start_rule_name, tokenizer)

    def update_state_with_batch_token_seqs(
        self, input_ids, batch_parsing_states, valid_token_start_idx=None
    ):

        if self.last_size is None:
            valid_prefix_tokens = [
                (
                    single_input_ids[valid_token_start_idx:]
                    if valid_token_start_idx is not None
                    else []
                )
                for single_input_ids in input_ids
            ]

            # self.grammar_acceptor.accept_token_ids(valid_prefix_tokens, self.stacks)
            resulting_batch_parsing_states = [
                self._update_state_with_single_token_seq(token_ids, parsing_state)
                for token_ids, parsing_state in zip(
                    valid_prefix_tokens, batch_parsing_states
                )
            ]
            #  if the length of the current input IDs (input_ids[0]) is exactly one more than self.last_size.
            #  This is expected in a scenario where inputs are processed incrementally, one token at a time.
            self.last_size = len(input_ids[0])
        else:

            # loop over the input_ids after the last_size
            resulting_batch_parsing_states = []

            for single_input_ids, _ in zip(input_ids, batch_parsing_states):
                valid_input_ids = single_input_ids[self.last_size :]
                # for i, token_id in enumerate(valid_input_ids):
                # parsing_state = self._consume_token_id(token_id, parsing_state)
                parsing_state = self._update_state_with_single_token_seq(
                    valid_input_ids, parsing_state=None
                )
                if len(parsing_state.stacks) == 0:
                    raise ValueError("The input is not accepted")
                resulting_batch_parsing_states.append(parsing_state)

        return resulting_batch_parsing_states


if __name__ == "__main__":
    from transformers import AutoTokenizer

    with open("examples/grammars/japanese.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)
    parsed_grammar.print()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenRecognizer = IncrementalTokenRecognizer(
        grammar_str=input_text, start_rule_name="root", tokenizer=tokenizer
    )

    japanese = "トリーム"  # "こんにちは"
    token_ids = tokenizer.encode(japanese)
    # 13298, 12675, 12045, 254
    init_state = None
    state = tokenRecognizer._update_state_with_single_token_seq(
        token_ids, init_state, as_string=False
    )

    if state.stacks:
        print("The Japanese input is accepted")
    else:
        print("The Japanese input is not accepted")

    korean = "안녕하세요"
    token_ids = tokenizer.encode(korean)
    init_state = tokenRecognizer.string_recognizer.get_initial_parsing_state()

    try:
        state = tokenRecognizer._update_state_with_single_token_seq(
            token_ids,
            init_state,
            as_string=False,
        )
        if state.stacks:
            print("The Korean input is accepted")
        else:
            print("The Korean input is not accepted")
    except ValueError as e:
        print("The Korean input is not accepted")
