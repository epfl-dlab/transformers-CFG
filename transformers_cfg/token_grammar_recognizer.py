import logging
from abc import ABC
from functools import lru_cache
from typing import List, Optional

import torch

from transformers_cfg.recognizer import StringRecognizer, AcceptState
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.tokenization.byte_trie import ByteTrie
from transformers_cfg.tokenization.middle.TokenizerMiddleMapping import (
    TokenizerMiddleMapping,
)

logger = logging.getLogger(__name__)


class AbsTokenRecognizer(ABC):
    def __init__(
        self,
        grammar_str,
        tokenizer,
        start_rule_name="root",
        trie=None,
        homomorphism=None,
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
        if homomorphism is None:
            self.homomorphism = TokenizerMiddleMapping.from_hf_tokenizer(tokenizer)
        else:
            self.homomorphism = homomorphism

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
        bytes_or_codepoints = self.homomorphism.map(token_id, verbose=False)
        new_acc_state = self.string_recognizer._update_state_with_bytes(
            bytes_or_codepoints, parsing_state, verbose=False
        )
        return len(new_acc_state.stacks) > 0

    def update_state_with_batch_token_seqs(self, *args, **kwargs):
        """Process a list of tokens according to the grammar rules."""
        raise NotImplementedError

    def batch_filter_vocab(self, batch_parsing_states, device) -> torch.Tensor:
        batch_acceptance = []
        for parsing_state in batch_parsing_states:
            batch_acceptance.append(self.filter_vocab(parsing_state, device))
        return torch.stack(batch_acceptance)

    def filter_vocab(self, parsing_state, device) -> torch.Tensor:
        if not parsing_state.stacks:  # Check if stacks is empty
            # Handle the empty case: for example, return a tensor of False
            # The size of the tensor should match the size of your vocabulary
            vocab_size = len(self.homomorphism)
            logger.debug(f"Empty stack, sum of acceptance: {0}")
            # size of the vocab
            accepts = [False] * vocab_size
            accepts[self.eos_token_id] = True
            return torch.tensor(accepts, dtype=torch.bool, device=device)

        acceptance = self.get_next_token_acceptance(parsing_state, device)

        return acceptance

    def get_next_token_acceptance(self, parsing_state, device) -> torch.Tensor:
        raise NotImplementedError

    def validate_and_set_eos_acceptance(self, acceptance: torch.Tensor) -> torch.Tensor:
        if torch.any(acceptance) == 0:
            acceptance[self.eos_token_id] = True
        else:
            if acceptance[self.eos_token_id]:
                raise ValueError()
            acceptance[self.eos_token_id] = False
        return acceptance

    def accept_token_ids(self, token_ids, stacks) -> bool:
        """Accept a list of token IDs according to the grammar rules."""
        raise NotImplementedError

    @staticmethod
    def detect_unicode(text: str) -> bool:
        # check if the text contains any unicode characters
        return any(ord(char) > 127 for char in text)


class IncrementalTokenRecognizer(AbsTokenRecognizer):
    def __init__(
        self, grammar_str, start_rule_name, tokenizer, trie=None, homomorphism=None
    ):
        super().__init__(
            grammar_str,
            tokenizer,
            start_rule_name,
            trie=trie,
            homomorphism=homomorphism,
        )
        self.last_size = None

    def _update_state_with_token_id(
        self, token_id: int, parsing_state: AcceptState
    ) -> AcceptState:
        if parsing_state.must_stop():
            if token_id == self.eos_token_id:
                return self.string_recognizer.get_termination_parsing_state()
            else:
                raise ValueError(
                    f"All stacks are empty, so the only token accepted is EOS({self.eos_token_id}), but got {token_id}.\
                        This error is likely due to the previous token not being accepted by the grammar."
                )
        if token_id == self.eos_token_id:
            if parsing_state.can_stop():
                # if at least one of the stack is empty, we can stop
                # we clear all the stacks, meaning that we don't accept any token after EOS
                return self.string_recognizer.get_termination_parsing_state()
            else:
                raise ValueError(
                    f"At least one of the stack should be empty when EOS is reached. However, "
                    f"the stacks are {parsing_state.stacks}"
                )

        bytes_or_codepoints = self.homomorphism.map(token_id)
        parsing_state = self.string_recognizer._update_state_with_bytes(
            bytes_or_codepoints, parsing_state
        )
        return parsing_state

        # if self.last_size is not set (which would be the case when processing the first token).
        # In this case, do nothing.

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
            batch_parsing_states = [
                self._update_state_with_single_token_seq(prefix, parsing_state)
                for prefix, parsing_state in zip(
                    valid_prefix_tokens, batch_parsing_states
                )
            ]
            #  if the length of the current input IDs (input_ids[0]) is exactly one more than self.last_size.
            #  This is expected in a scenario where inputs are processed incrementally, one token at a time.
        elif len(input_ids[0]) == self.last_size + 1:
            batch_parsing_states = [
                self._update_state_with_token_id(single_input_ids[-1], parsing_state)
                for single_input_ids, parsing_state in zip(
                    input_ids, batch_parsing_states
                )
            ]
            #  ensure that the input size is consistent with the expected incremental processing
            #  (i.e., one token at a time).
        else:
            # here we check if the input_ids are one token longer than the last time we processed
            # but we don't check if input_ids are actually valid.
            # Imagine a scenario where we generate 10 tokens, then we replace the 10 generated tokens with 10 new tokens.
            # In this case, the input_ids will be consistent with the last_size, but the input_ids are not valid.
            # However, should we really check if the input_ids are valid here?
            # If we do, then we need to reparse the whole input_ids at each call, which is not efficient.
            # Maybe we should just trust the user to provide valid input_ids?
            # The conclusion is that, we assume the input_ids are valid, and our generation will be correct.
            # If the input_ids are not valid, then the generation result will be wrong and we don't take responsibility for that.
            raise RuntimeError(
                "Input ID's length is inconsistent with the current state of "
                "the GrammarConstrainedLogitsProcessor. If you want to process "
                "another input sequence, please instantiate a new "
                "GrammarConstrainedLogitsProcessor."
            )
        self.last_size = len(input_ids[0])

        return batch_parsing_states

    def _update_state_with_single_token_seq(
        self,
        token_ids: List[int],
        parsing_state: Optional[AcceptState] = None,
        as_string=True,
    ):
        if parsing_state is None:
            parsing_state = self.string_recognizer.get_initial_parsing_state()
        if as_string:
            string = self.tokenizer.decode(token_ids)
            parsing_state = self.string_recognizer._update_state_with_string(
                string, parsing_state
            )
        else:
            for i, token_id in enumerate(token_ids):
                parsing_state = self._update_state_with_token_id(
                    token_id, parsing_state
                )
                if len(parsing_state.stacks) > 0:
                    cur_token_ids = token_ids[: i + 1]
                    logging.debug(f"{cur_token_ids} is accepted")
                    decoded_string = self.tokenizer.decode(cur_token_ids)
                    logging.debug(f"The decoded string is {decoded_string}")
        return parsing_state

    def accept_token_ids(self, token_ids, stacks=None, as_string=True) -> bool:
        output_state = self._update_state_with_single_token_seq(
            token_ids, stacks, as_string
        ).stacks
        return True if output_state else False

    def get_next_token_acceptance(self, parsing_state, device) -> torch.Tensor:
        acceptance_matrix = torch.cat(
            [
                self.get_next_token_acceptance_for_single_stack(
                    tuple(stack), parsing_state.partial_utf8, device
                )
                for stack in parsing_state.stacks
            ]
        )
        # Merge stacks: any True => True
        acceptance = acceptance_matrix.reshape(len(parsing_state.stacks), -1).any(dim=0)
        return acceptance

    # If running on a GPU device this cache can continue to fill up GPU memory
    # Dereferencing the object will not clear the cache
    @lru_cache(maxsize=32768)
    def get_next_token_acceptance_for_single_stack(self, stack, partial_utf8, device):
        # stack = list(stack)  # needs to come in as a tuple for lru_cache
        assert isinstance(stack, tuple)

        if self.use_unicode:

            accept_f = lambda x: self.string_recognizer._try_accept_bytes(
                x, {stack}, partial_utf8=partial_utf8
            )
            token_acceptance = self.byte_trie.get_next_token_acceptance(
                accept=accept_f, accept_eos=False, eos_token_id=self.eos_token_id
            )
        else:
            accepts = [False] * len(self.homomorphism)
            token_acceptance = check_token_acceptance_in_trie(
                self.byte_trie.root,
                [stack],
                self.string_recognizer,
                self.eos_token_id,
                accepts,
            )
        x = torch.tensor(token_acceptance, dtype=torch.bool, device=device)
        x_eos = self.validate_and_set_eos_acceptance(x)
        return x_eos


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


def check_token_acceptance_in_trie(trie_node, stacks, grammar, eos_token_id, accepts):
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
            num_chars = grammar.grammar_encoding[next_element_offset]

            if not grammar.char_acceptance_at_element(next_element_offset).get(
                byte, False
            ):
                # if the current byte is not accepted by the current rule, we need to try next rule
                continue

            next_element_offset += num_chars + 1
            new_stack = list(stk[:-1])
            if grammar.grammar_encoding[next_element_offset]:
                new_stack.append(next_element_offset)
            new_stacks.update(grammar.expand_stack_head(tuple(new_stack)))

        if new_stacks:
            check_token_acceptance_in_trie(
                next_trie_node, new_stacks, grammar, eos_token_id, accepts
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
