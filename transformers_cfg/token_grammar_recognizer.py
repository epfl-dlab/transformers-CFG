import copy
import logging
from abc import ABC
from functools import lru_cache
from typing import List

import torch

from transformers_cfg.recognizer import GrammarRecognizer, AcceptState
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.trie import Trie
from transformers_cfg.utf8_utils import PartialUTF8
from .vocab_struct import LEAF, TokenTrie
from transformers_cfg.mapping import get_mapping

logger = logging.getLogger(__name__)


def check_token_acceptance_in_trie(
    trie, stacks, grammar, eos_token_id, accepts, partial_utf8
):

    for byte, next_trie in trie.items():
        if byte == LEAF:
            token_id = next_trie
            if token_id != eos_token_id:
                # if the stacks is not empty, it means we can still continue to parse
                # so we should accept the token
                accepts[token_id] = bool(stacks)
            continue

        new_stacks = []
        for stk in stacks:
            if not stk:
                continue

            next_element_offset = stk[-1]
            num_chars = grammar.grammar_encoding[next_element_offset]

            if not grammar._consume_byte_partial_match():
                # if the current byte is not accepted by the current rule, we need to try next rule
                continue

            next_element_offset += num_chars + 1
            new_stack = stk[:-1]
            if grammar.grammar_encoding[next_element_offset]:
                new_stack.append(next_element_offset)
            new_stacks.extend(grammar.advance_stack(tuple(new_stack)))

        if new_stacks:
            check_token_acceptance_in_trie(
                next_trie, new_stacks, grammar, eos_token_id, accepts
            )

    return accepts


class AbsTokenGrammarRecognizer(ABC):
    def __init__(self, grammar_str, tokenizer, start_rule_name="root"):
        parsed_grammar = parse_ebnf(grammar_str)
        grammar_encoding = parsed_grammar.grammar_encoding
        self.start_rule_id = parsed_grammar.symbol_table.get(start_rule_name)

        self.eos_token_id = tokenizer.eos_token_id
        self.mapping = get_mapping(tokenizer)
        self.token_trie = TokenTrie(tokenizer)
        self.tokenizer = tokenizer
        assert len(self.mapping) == len(
            self.token_trie
        ), f"{len(self.mapping)}, {len(self.token_trie)}"
        self.grammar = GrammarRecognizer(grammar_encoding, self.start_rule_id)
        # TODO rename to self.recognizer
        self.trie = Trie.from_tokenizer(tokenizer)
        # self.partial_utf8 = PartialUTF8()

    def _consume_token_id(
        self, token_id: int, accept_state: AcceptState
    ) -> AcceptState:
        if self.grammar._must_stop(accept_state.stacks):
            if token_id == self.eos_token_id:
                return self.grammar.init_accept_state()
            else:
                raise ValueError(
                    f"All stacks are empty, so the only token accepted is EOS({self.eos_token_id}), but got {token_id}"
                )
        if token_id == self.eos_token_id:
            if self.grammar._can_stop(accept_state.stacks):
                # if at least one of the stack is empty, we can stop
                # we clear all the stacks, meaning that we don't accept any token after EOS
                return self.grammar.init_accept_state()
            else:
                raise ValueError(
                    f"At least one of the stack should be empty when EOS is reached. However, "
                    f"the stacks are {accept_state.stacks}"
                )
        # for code_point in self.mapping.map(token_id):
        #     stacks = self.grammar._consume_char_code_point(code_point, stacks)
        bytes_or_codepoints = self.mapping.map(token_id)
        accept_state = self.grammar._consume_bytes_partial_match(
            bytes_or_codepoints, accept_state
        )
        return accept_state

    def probe_token_id(self, token_id: int, accept_state: AcceptState) -> bool:
        stacks = accept_state.stacks
        if self.grammar._must_stop(stacks):
            if token_id == self.eos_token_id:
                return True
            else:
                return False
        if token_id == self.eos_token_id:
            if self.grammar._can_stop(stacks):
                # if at least one of the stack is empty, we can stop
                # we clear all the stacks, meaning that we don't accept any token after EOS
                return True
            else:
                return False
        # for code_point in self.mapping.map(token_id):
        #     stacks = self.grammar._consume_char_code_point(code_point, stacks)
        bytes_or_codepoints = self.mapping.map(token_id, verbose=False)
        new_acc_state = self.grammar._consume_bytes_partial_match(
            bytes_or_codepoints, accept_state, verbose=False
        )
        return len(new_acc_state.stacks) > 0

    def advance_token_ids(self, *args, **kwargs):
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
            vocab_size = len(self.mapping)
            logger.debug(f"Empty stack, sum of acceptance: {0}")
            # size of the vocab
            accepts = [False] * vocab_size
            accepts[self.eos_token_id] = True
            return torch.tensor(accepts, dtype=torch.bool, device=device)

        mode = "trie"

        # TODO for debugging
        # if mode == "lru":

        acceptance_matrix = torch.cat(
            [
                self.token_acceptance_for_stack(
                    tuple(stack), accept_state.partial_utf8, device
                )
                for stack in accept_state.stacks
            ]
        )
        # Merge stacks: any True => True
        # acceptance_3 = acceptance_matrix.reshape(len(accept_state.stacks), -1).any(dim=0)
        # elif mode == "trie":
        acceptance_2 = self.check_token_acceptance(accept_state, device)
        # # else:
        #     acceptance_1 = self.check_token_acceptance_old(accept_state, device)
        #     assert torch.equal(acceptance_1, acceptance_2), f"acceptance_1: {acceptance_1}, acceptance_2: {acceptance_2}"
        # assert torch.equal(acceptance_2, acceptance_3), f"acceptance_2: {acceptance_2}, acceptance_3: {acceptance_3}"

        # logger.debug(f"sum of acceptance: {acceptance.sum()}")
        return acceptance_2

    # Probably this should be configurable. If the grammar has an exceedingly
    # large number of states, the correct setting is a tradeoff between GPU
    # RAM usage and recomputation time.
    #
    # The main variable that pushes usage up here is number of states in the
    # grammar.
    # @lru_cache(maxsize=32768)
    # def token_acceptance_for_stack(self, stack, device):
    #     stack = list(stack)  # needs to come in as a tuple for lru_cache
    #
    #     # size of the vocab
    #     accepts = [False] * len(self.mapping)
    #     if len(stack) == 0:
    #         logger.debug("empty stack")
    #         accepts[self.eos_token_id] = len(stack) == 0
    #     else:
    #         accepts = check_token_acceptance_in_trie(
    #             self.tokenizer.vocab.keys(),
    #             [stack],
    #             self.grammar,
    #             self.eos_token_id,
    #         )
    #         if sum(accepts) == 0:
    #             # if no token is accepted, we accept EOS
    #             accepts[self.eos_token_id] = True
    #     x = torch.tensor(accepts, dtype=torch.bool, device=device)
    #     return x

    def check_token_acceptance_old(self, accept_state, device):
        vocab = self.tokenizer.vocab.values()
        accepts: List[bool] = [False] * len(vocab)
        if not accept_state.stacks:
            accepts[self.eos_token_id] = True
            return torch.tensor(accepts, dtype=torch.bool, device=device)

        for token_id in vocab:
            accepts[token_id] = self.probe_token_id(token_id, accept_state)
        if sum(accepts) == 0:
            accepts[self.eos_token_id] = True

        x = torch.tensor(accepts, dtype=torch.bool, device=device)
        return x

    def check_token_acceptance(self, accept_state, device):

        vocab = self.tokenizer.vocab.values()
        accepts: List[bool] = [False] * len(vocab)
        if not accept_state.stacks:
            accepts[self.eos_token_id] = True
            return torch.tensor(accepts, dtype=torch.bool, device=device)

        partial_utf8 = accept_state.partial_utf8
        stacks = accept_state.stacks
        accepts: List[bool] = self.trie.get_token_acceptance(
            accept=lambda x: self.grammar._probe_bytes_partial_match(
                x, stacks, partial_utf8
            ),
            accept_eos=False,
            eos_token_id=self.eos_token_id,
        )

        x = torch.tensor(accepts, dtype=torch.bool, device=device)
        x_eos = self._check_eos_token_acceptance(x)
        return x_eos

    @lru_cache(maxsize=32768)
    def token_acceptance_for_stack(self, stack, partial_utf8, device):
        # stack = list(stack)  # needs to come in as a tuple for lru_cache
        assert isinstance(stack, tuple)
        stack = list(stack)

        # token_acceptance = check_token_acceptance_in_trie(
        #     self.trie, [stack], self.grammar, partial_utf8, accept_eos=False, eos_token_id=self.eos_token_id
        # )
        accept_f = lambda x: self.grammar._probe_bytes_partial_match(
            x, [stack], partial_utf8=partial_utf8
        )
        token_acceptance = self.trie.get_token_acceptance(
            accept=accept_f, accept_eos=False, eos_token_id=self.eos_token_id
        )
        x = torch.tensor(token_acceptance, dtype=torch.bool, device=device)
        x_eos = self._check_eos_token_acceptance(x)
        return x_eos

    def _check_eos_token_acceptance(self, acceptance: torch.Tensor) -> torch.Tensor:
        if any(acceptance) == 0:
            acceptance[self.eos_token_id] = True
        else:
            if acceptance[self.eos_token_id]:
                raise ValueError()
            acceptance[self.eos_token_id] = False
        return acceptance


class IncrementalTokenGrammarRecognizer(AbsTokenGrammarRecognizer):
    def __init__(self, grammar_str, start_rule_name, tokenizer):
        super().__init__(grammar_str, tokenizer, start_rule_name)
        self.last_size = None

        # if self.last_size is not set (which would be the case when processing the first token).
        # In this case, do nothing.

    def advance_token_ids(self, input_ids, batch_accept_states, parse_start_index=None):

        if self.last_size is None:
            prefix_to_parse = [
                single_input_ids[parse_start_index:]
                if parse_start_index is not None
                else []
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
        self, token_ids: List[int], accept_state: AcceptState = None, as_string=True
    ):
        if accept_state is None:
            accept_state = self.grammar.init_accept_state()
        if as_string:
            string = self.tokenizer.decode(token_ids)
            accept_state = self.grammar._consume_string(string, accept_state)
        else:
            for i, token_id in enumerate(token_ids):
                accept_state = self._consume_token_id(token_id, accept_state)
                if len(accept_state.stacks) > 0:
                    cur_token_ids = token_ids[: i + 1]
                    logging.debug(f"{cur_token_ids} is accepted")
                    decoded_string = self.tokenizer.decode(cur_token_ids)
                    logging.debug(f"The decoded string is {decoded_string}")
        return accept_state


class VanillaTokenGrammarRecognizer(AbsTokenGrammarRecognizer):
    # TODO, this class may have been broken because of the recent refactoring, in particular, the stacks is bound to the
    # parser class
    def __init__(self, grammar_str, start_rule_name, tokenizer):
        super().__init__(grammar_str, tokenizer, start_rule_name)
        self.offset = None

    def advance_token_ids(self, input_ids, batch_stacks, parse_start_index=None):
        # By design, the batch_stacks should be empty at the beginning, thus it doesn't matter what we pass in.
        if self.offset is None:
            self.offset = (
                len(input_ids[0]) if parse_start_index is None else parse_start_index
            )

        # TODO: here may be broken, the deepcopy may not work
        batch_stacks_from_scratch = [
            copy.deepcopy(self.grammar.stacks) for _ in range(len(input_ids))
        ]

        prefix_to_consume = [
            single_input_ids[self.offset :] for single_input_ids in input_ids
        ]
        # self.grammar_acceptor.accept_token_ids(prefix_to_consume, self.stacks)
        advanced_batch_stacks = [
            self._consume_token_ids(prefix, stack, as_string=False)
            for prefix, stack in zip(prefix_to_consume, batch_stacks_from_scratch)
        ]
        return advanced_batch_stacks

    def _consume_token_ids(
        self, token_ids: List[int], accept_state: AcceptState, as_string=True
    ):
        if as_string:
            string = self.tokenizer.decode(token_ids)
            accept_state = self.grammar._consume_string(string, accept_state)
        else:
            for token_id in token_ids:
                accept_state = self._consume_token_id(token_id, accept_state)
        return accept_state


# if __name__ == "__main__":
#     from transformers import AutoTokenizer
#
#     with open("examples/grammars/json.ebnf", "r") as file:
#         input_text = file.read()
#     parsed_grammar = parse_ebnf(input_text)
#     parsed_grammar.print()
#
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#
#     tokenRecognizer = IncrementalTokenGrammarRecognizer(
#         grammar_str=input_text, start_rule_name="root", tokenizer=tokenizer
#     )
#
#     valid_json = '{"foo": "bar", "baz": "bat"}'
#     token_ids = tokenizer.encode(valid_json)
#     stacks = tokenRecognizer._consume_token_ids(
#         token_ids, tokenRecognizer.grammar.stacks, as_string=False
#     )
#     # the json object is complete, so the stacks should be empty
#     assert stacks == [] or stacks == [[]], f"stacks: {stacks}, not empty"

if __name__ == "__main__":
    from transformers import AutoTokenizer

    with open("examples/grammars/japanese.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)
    parsed_grammar.print()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenRecognizer = IncrementalTokenGrammarRecognizer(
        grammar_str=input_text, start_rule_name="root", tokenizer=tokenizer
    )

    japanese = "トリーム"  # "こんにちは"
    token_ids = tokenizer.encode(japanese)
    # 13298, 12675, 12045, 254
    stacks = tokenRecognizer._consume_token_ids(
        token_ids, tokenRecognizer.grammar.stacks, as_string=False
    )

    if stacks:
        print("The Japanese input is accepted")
    else:
        print("The Japanese input is not accepted")

    korean = "안녕하세요"
    token_ids = tokenizer.encode(korean)

    try:
        stacks = tokenRecognizer._consume_token_ids(
            token_ids, tokenRecognizer.grammar.stacks, as_string=False
        )
        if stacks:
            print("The Korean input is accepted")
        else:
            print("The Korean input is not accepted")
    except ValueError as e:
        print("The Korean input is not accepted")
