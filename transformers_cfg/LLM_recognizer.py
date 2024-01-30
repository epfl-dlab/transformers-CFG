import copy
import logging
from abc import ABC
from functools import lru_cache
from typing import List

import torch

from transformers_cfg.recognizer import GrammarRecognizer
from transformers_cfg.parser import parse_ebnf
from .vocab_struct import LEAF, TokenTrie

logger = logging.getLogger(__name__)


class AbsLLMGrammarRecognizer(ABC):
    def __init__(self, grammar_str, tokenizer, start_rule_name="root"):
        parsed_grammar = parse_ebnf(grammar_str)
        grammar_encoding = parsed_grammar.grammar_encoding
        self.start_rule_id = parsed_grammar.symbol_table.get(start_rule_name)

        self.eos_token_id = tokenizer.eos_token_id
        self.token_trie = TokenTrie(tokenizer)
        self.tokenizer = tokenizer
        self.grammar = GrammarRecognizer(grammar_encoding, self.start_rule_id)

    def _consume_token_id(self, token_id: int, stacks: List[List[int]]):
        if token_id == self.eos_token_id:
            if stacks and all(len(stack) != 0 for stack in stacks):
                raise Exception(
                    f"At least one of the stack should be empty when EOS is reached. However, "
                    f"the stacks are {stacks}"
                )
            return []

        for byte in self.token_trie.id2str(token_id):
            stacks = self.grammar._consume_char(byte, stacks)
            # check updated stacks
            # TODO, I commented this out because it will fail when the stack is empty
            # empty stack means the end of the grammar
            # assert stacks != []

        return stacks

    def advance_token_ids(self, *args, **kwargs):
        """Process a list of tokens according to the grammar rules."""
        raise NotImplementedError

    def batch_filter_vocab(self, batch_stacks, device) -> torch.Tensor:
        batch_acceptance = []
        for stacks in batch_stacks:
            batch_acceptance.append(self.filter_vocab(stacks, device))
        return torch.stack(batch_acceptance)

    def filter_vocab(self, stacks, device) -> torch.Tensor:
        if not stacks:  # Check if stacks is empty
            # Handle the empty case: for example, return a tensor of False
            # The size of the tensor should match the size of your vocabulary
            vocab_size = len(self.token_trie)
            logger.debug(f"Empty stack, sum of acceptance: {0}")
            return torch.zeros(vocab_size, dtype=torch.bool, device=device)

        acceptance_matrix = torch.cat(
            [self.token_acceptance_for_stack(tuple(stack), device) for stack in stacks]
        )
        # Merge stacks: any True => True
        acceptance = acceptance_matrix.reshape(len(stacks), -1).any(dim=0)
        logger.debug(f"sum of acceptance: {acceptance.sum()}")
        return acceptance

    # For each sub-rule in the grammar, cache whether each byte is accepted.
    @lru_cache(maxsize=None)
    def char_acceptance_at_rule_pos(self, rule_offset):
        # every time this function is called, the result is cached
        # next time when the same pos is called, the result is returned directly
        # Here the pos corresponds to the literal or char range rule
        # it doesn't handle the rule reference
        acceptance = [False] * 256
        num_chars = self.grammar.grammar_encoding[rule_offset]
        rule_offset += 1
        for i in range(0, num_chars, 2):
            start = self.grammar.grammar_encoding[rule_offset + i]
            end = self.grammar.grammar_encoding[rule_offset + i + 1]
            for j in range(start, end + 1):
                acceptance[j] = True
        return acceptance

    # Probably this should be configurable. If the grammar has an exceedingly
    # large number of states, the correct setting is a tradeoff between GPU
    # RAM usage and recomputation time.
    #
    # The main variable that pushes usage up here is number of states in the
    # grammar.
    @lru_cache(maxsize=32768)
    def token_acceptance_for_stack(self, stack, device):
        stack = list(stack)  # needs to come in as a tuple for lru_cache

        accepts = [False] * len(self.token_trie)
        accepts[self.eos_token_id] = len(stack) == 0
        if len(stack) == 0:
            logger.debug("empty stack")

        def traverse_trie(trie, stacks):
            for byte, next_trie in trie.items():
                if byte == LEAF:
                    token_id = next_trie
                    if token_id != self.eos_token_id:
                        # if the stacks is not empty, it means we can still continue to parse
                        # so we should accept the token
                        accepts[token_id] = bool(stacks)
                    continue

                new_stacks = []
                for stk in stacks:
                    if not stk:
                        continue

                    next_rule_pos = stk[-1]
                    num_chars = self.grammar.grammar_encoding[next_rule_pos]

                    if not self.char_acceptance_at_rule_pos(next_rule_pos)[byte]:
                        # if the current byte is not accepted by the current rule, we need to try next rule
                        continue

                    next_rule_pos += num_chars + 1
                    new_stack = stk[:-1]
                    if self.grammar.grammar_encoding[next_rule_pos]:
                        new_stack.append(next_rule_pos)
                    new_stacks.extend(self.grammar.advance_stack(tuple(new_stack)))

                if new_stacks:
                    traverse_trie(next_trie, new_stacks)

        traverse_trie(self.token_trie.trie, [stack])

        x = torch.tensor(accepts, dtype=torch.bool, device=device)
        return x


class IncrementalLLMGrammarRecognizer(AbsLLMGrammarRecognizer):
    def __init__(self, grammar_str, start_rule_name, tokenizer):
        super().__init__(grammar_str, tokenizer, start_rule_name)
        self.last_size = None

        # if self.last_size is not set (which would be the case when processing the first token).
        # In this case, do nothing.

    def advance_token_ids(self, input_ids, batch_stacks, parse_start_index=None):

        if self.last_size is None:
            prefix_to_parse = [
                single_input_ids[parse_start_index:]
                if parse_start_index is not None
                else []
                for single_input_ids in input_ids
            ]

            # self.grammar_acceptor.accept_token_ids(prefix_to_parse, self.stacks)
            batch_stacks = [
                self._consume_token_ids(prefix, stack)
                for prefix, stack in zip(prefix_to_parse, batch_stacks)
            ]
            #  if the length of the current input IDs (input_ids[0]) is exactly one more than self.last_size.
            #  This is expected in a scenario where inputs are processed incrementally, one token at a time.
        elif len(input_ids[0]) == self.last_size + 1:
            batch_stacks = [
                self._consume_token_id(single_input_ids[-1], stack)
                for single_input_ids, stack in zip(input_ids, batch_stacks)
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

        return batch_stacks

    def _consume_token_ids(
        self, token_ids: List[int], stacks: List[List[int]], as_string=True
    ):
        if as_string:
            string = self.tokenizer.decode(token_ids)
            stacks = self.grammar._consume_string(string, stacks)
        else:
            for token_id in token_ids:
                stacks = self._consume_token_id(token_id, stacks)
        return stacks


class VanillaLLMGrammarRecognizer(AbsLLMGrammarRecognizer):
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
        self, token_ids: List[int], stacks: List[List[int]], as_string=True
    ):
        if as_string:
            string = self.tokenizer.decode(token_ids)
            stacks = self.grammar._consume_string(string, stacks)
        else:
            for token_id in token_ids:
                stacks = self._consume_token_id(token_id, stacks)
        return stacks


if __name__ == "__main__":
    # set logging level
    logging.basicConfig(level=logging.DEBUG)

    try:
        with open("examples/grammars/json.ebnf", "r") as file:
            input_text = file.read()
        parsed_grammar = parse_ebnf(input_text)
        parsed_grammar.print()
        print(f"symbol_ids: \n{parsed_grammar.symbol_table}")

    except FileNotFoundError:
        print("Error: File 'grammar.ebnf' not found.")
    except IOError as e:
        print("Error reading file 'grammar.ebnf':", e)
