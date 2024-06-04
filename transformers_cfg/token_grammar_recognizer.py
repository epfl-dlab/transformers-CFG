import logging
from abc import ABC
from functools import lru_cache
from typing import List, Optional

import torch

from transformers_cfg.recognizer import StringRecognizer, AcceptState
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.tokenization.byte_trie import ByteTrie
from transformers_cfg.tokenization.codepoint_trie import LEAF, CodePointTrie
from transformers_cfg.tokenization.mapping import get_mapping
from typing import Set, Tuple

logger = logging.getLogger(__name__)


class AbsTokenRecognizer(ABC):
    def __init__(self, grammar_str, tokenizer, start_rule_name="root", unicode=False):
        parsed_grammar = parse_ebnf(grammar_str)
        grammar_encoding = parsed_grammar.grammar_encoding
        self.start_rule_id = parsed_grammar.symbol_table.get(start_rule_name)
        self.byte_encoding = unicode

        if unicode and not tokenizer.__class__.__name__.lower().startswith(
            "gpt2"
        ):  # gpt2tokenizer or gpt2tokenizerfast
            raise ValueError(
                "Constrained decoding with unicode is only supported for GPT2 model. Support for other models is coming soon."
                "Or you can use the constraints with only ascii characters."
            )

        self.eos_token_id = tokenizer.eos_token_id
        self.code_point_token_trie = CodePointTrie(tokenizer)
        self.tokenizer = tokenizer
        self.string_recognizer = StringRecognizer(grammar_encoding, self.start_rule_id)
        self.unicode_trie = ByteTrie.from_tokenizer(tokenizer, unicode=unicode)
        self.mapping = get_mapping(tokenizer, unicode=unicode)
        self.vocab_size = len(self.mapping)
        assert self.vocab_size == len(
            self.code_point_token_trie
        ), f"{self.vocab_size}, {len(self.code_point_token_trie)}"

    def _must_stop(self, stacks: Set[Tuple[int]]):
        return len(stacks) == 0 or all(len(stack) == 0 for stack in stacks)

    def _can_stop(self, stacks: Set[Tuple[int]]):
        # if at least one of the stack is empty, we can stop
        return len(stacks) == 0 or any(len(stack) == 0 for stack in stacks)

    def _consume_token_id(
        self, token_id: int, accept_state: AcceptState
    ) -> AcceptState:
        if self._must_stop(accept_state.stacks):
            if token_id == self.eos_token_id:
                self.mapping.last_token_id = None
                return AcceptState.empty_state()
            else:
                raise ValueError(
                    f"All stacks are empty, so the only token accepted is EOS({self.eos_token_id}), but got {token_id}"
                )
        if token_id == self.eos_token_id:
            if self._can_stop(accept_state.stacks):
                self.mapping.last_token_id = None
                # we clear all the stacks, meaning that we don't accept any token after EOS
                return AcceptState.empty_state()
            else:
                raise ValueError(
                    f"At least one of the stack should be empty when EOS is reached. However, "
                    f"the stacks are {accept_state.stacks}"
                )

        bytes_or_codepoints = self.mapping.map(token_id)
        accept_state = self.string_recognizer._consume_bytes(
            bytes_or_codepoints, accept_state
        )
        return accept_state

    def consume_token_ids(self, *args, **kwargs):
        """Process a list of tokens according to the grammar rules."""
        raise NotImplementedError

    def batch_filter_vocab(self, batch_accept_states, device) -> torch.Tensor:
        batch_acceptance = []
        for accept_state in batch_accept_states:
            batch_acceptance.append(self.filter_vocab(accept_state, device))
        return torch.stack(batch_acceptance)

    def filter_vocab(self, accept_state, device) -> torch.Tensor:
        if not accept_state.stacks:  # Check if stacks is empty
            # Handle the empty case: for example, return a tensor of False
            # The size of the tensor should match the size of your vocabulary
            logger.debug(f"Empty stack, sum of acceptance: {0}")
            # size of the vocab
            accepts = [False] * self.vocab_size
            accepts[self.eos_token_id] = True
            return torch.tensor(accepts, dtype=torch.bool, device=device)

        acceptance = self.get_token_acceptance(accept_state, device)

        return acceptance

    def get_token_acceptance(self, accept_state, device) -> torch.Tensor:
        acceptance_matrix = torch.cat(
            [
                self.get_token_acceptance_array_for_stack(
                    tuple(stack), accept_state.partial_utf8, device
                )
                for stack in accept_state.stacks
            ]
        )
        # Merge stacks: any True => True
        acceptance = acceptance_matrix.reshape(len(accept_state.stacks), -1).any(dim=0)
        return acceptance

    @lru_cache(maxsize=32768)
    def get_token_acceptance_array_for_stack(self, stack: Tuple, partial_utf8, device):

        assert isinstance(stack, tuple)

        token_acceptance = [False] * self.vocab_size

        if self.byte_encoding:
            # boolean function checking if a byte sequence is accepted by the grammar
            accept_f = lambda x: self.string_recognizer._try_accept_bytes(
                bytes(x), {stack}, partial_utf8=partial_utf8
            )
            self.unicode_trie.get_token_acceptance(
                accept=accept_f,
                accept_eos=False,
                eos_token_id=self.eos_token_id,
                token_acceptance=token_acceptance,
            )
        else:
            check_token_acceptance_in_trie(
                self.code_point_token_trie.trie,
                [stack],
                self.string_recognizer,
                self.eos_token_id,
                token_acceptance,
            )
        x = torch.tensor(token_acceptance, dtype=torch.bool, device=device)
        x_eos = self.validate_and_set_eos_acceptance(x)
        return x_eos

    def validate_and_set_eos_acceptance(self, acceptance: torch.Tensor) -> torch.Tensor:
        if torch.any(acceptance) == 0:
            acceptance[self.eos_token_id] = True
        else:
            if acceptance[self.eos_token_id]:
                raise ValueError()
            acceptance[self.eos_token_id] = False
        return acceptance


class IncrementalTokenRecognizer(AbsTokenRecognizer):
    def __init__(self, grammar_str, start_rule_name, tokenizer, unicode=False):
        super().__init__(grammar_str, tokenizer, start_rule_name, unicode)
        self.last_size = None

        # if self.last_size is not set (which would be the case when processing the first token).
        # In this case, do nothing.

    def consume_token_ids(self, input_ids, batch_accept_states, parse_start_index=None):

        if self.last_size is None:
            prefix_to_parse = [
                (
                    single_input_ids[parse_start_index:]
                    if parse_start_index is not None
                    else []
                )
                for single_input_ids in input_ids
            ]

            # self.grammar_acceptor.accept_token_ids(prefix_to_parse, self.stacks)
            batch_accept_states = [
                self._consume_token_ids(prefix, accept_state)
                for prefix, accept_state in zip(prefix_to_parse, batch_accept_states)
            ]
            #  if the length of the current input IDs (input_ids[0]) is exactly one more than self.last_size.
            #  This is expected in a scenario where inputs are processed incrementally, one token at a time.
        elif len(input_ids[0]) == self.last_size + 1:
            batch_accept_states = [
                self._consume_token_id(single_input_ids[-1], accept_state)
                for single_input_ids, accept_state in zip(
                    input_ids, batch_accept_states
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

        return batch_accept_states

    def _consume_token_ids(
        self,
        token_ids: List[int],
        accept_state: Optional[AcceptState] = None,
        as_string=True,
    ):
        if accept_state is None:
            accept_state = self.string_recognizer.get_initial_accept_state()
        if as_string:
            string = self.tokenizer.decode(token_ids)
            accept_state = self.string_recognizer._consume_string(string, accept_state)
        else:
            print(self.tokenizer.eos_token_id in token_ids)
            for i, token_id in enumerate(token_ids):
                accept_state = self._consume_token_id(token_id, accept_state)
                if len(accept_state.stacks) > 0:
                    cur_token_ids = token_ids[: i + 1]
                    logging.debug(f"{cur_token_ids} is accepted")
                    decoded_string = self.tokenizer.decode(cur_token_ids)
                    logging.debug(f"The decoded string is {decoded_string}")
        return accept_state


def check_token_acceptance_in_trie(trie, stacks, grammar, eos_token_id, accepts):
    for byte, next_trie in trie.items():
        if byte == LEAF:
            token_id = next_trie
            if token_id != eos_token_id:
                # if the stacks is not empty, it means we can still continue to parse
                # so we should accept the token
                accepts[token_id] = bool(stacks)
            continue

        new_stacks = set()
        for stack in stacks:
            if not stack:
                continue

            next_element_offset = stack[-1]
            num_chars = grammar.grammar_encoding[next_element_offset]

            # if the current byte is not accepted by the current rule, we need to try next rule
            if not grammar.accept_code_point_at_element(byte, next_element_offset):
                continue

            next_element_offset += num_chars + 1
            new_stack = list(stack[:-1])
            if grammar.grammar_encoding[next_element_offset]:
                new_stack.append(next_element_offset)
            new_stacks.update(grammar.expand_stack_head(tuple(new_stack)))

        if new_stacks:
            check_token_acceptance_in_trie(
                next_trie, new_stacks, grammar, eos_token_id, accepts
            )

    return accepts


if __name__ == "__main__":
    from transformers import AutoTokenizer

    with open("examples/grammars/japanese.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)
    parsed_grammar.print()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenRecognizer = IncrementalTokenRecognizer(
        grammar_str=input_text,
        start_rule_name="root",
        tokenizer=tokenizer,
        unicode=True,
    )

    japanese = "トリーム"  # "こんにちは"
    token_ids = tokenizer.encode(japanese)
    # 13298, 12675, 12045, 254
    init_state = None
    state = tokenRecognizer._consume_token_ids(token_ids, init_state, as_string=False)

    if state.stacks:
        print("The Japanese input is accepted")
    else:
        print("The Japanese input is not accepted")

    korean = "안녕하세요"
    token_ids = tokenizer.encode(korean)
    init_state = tokenRecognizer.string_recognizer.get_initial_accept_state()

    try:
        state = tokenRecognizer._consume_token_ids(
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
