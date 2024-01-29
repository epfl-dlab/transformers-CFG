import logging
import sys
import copy
from functools import lru_cache
from typing import List, Tuple

from transformers_cfg.parsing import (
    END_OF_RULE_MARKER,
    END_OF_ALTERNATE_MARKER,
    parse_ebnf,
    print_grammar,
)


class LlamaGrammar:
    def __init__(self, grammar_encoding: List[int], start_rule_id: int):
        rule_offset = 0
        rule_offsets = {}
        self.grammar_encoding = copy.deepcopy(grammar_encoding)

        # Build `rules` as a dictionary of rule IDs to their positions in `grammar_src`
        while self.grammar_encoding[rule_offset] != 0xFFFF:
            rule_id = self.grammar_encoding[rule_offset]
            # store the offset idx
            rule_offsets[rule_id] = rule_offset

            # Skip rule ID
            # rule_offset += 1
            simple_rhs_offset = rule_offset + 1

            # Skip rule alternates
            while self.grammar_encoding[simple_rhs_offset] != END_OF_RULE_MARKER:
                simple_rhs_offset = (
                    simple_rhs_offset + 1 + self.grammar_encoding[simple_rhs_offset]
                )

            # Skip 0 denoting end of rule
            # rule_offset += 1
            rule_offset = simple_rhs_offset + 1

        retrieved_start_rule_id = self.grammar_encoding[rule_offsets[start_rule_id]]
        assert retrieved_start_rule_id == start_rule_id

        # Loop over alternates of start rule to build initial stacks
        sub_rhs_offset = rule_offsets[start_rule_id] + 1
        stacks = []
        while self.grammar_encoding[sub_rhs_offset]:
            stack: List[int] = []
            if self.grammar_encoding[sub_rhs_offset + 1] != END_OF_ALTERNATE_MARKER:
                # If alternate is nonempty, add to stack
                stack.append(self.grammar_encoding[sub_rhs_offset + 1])
            stacks.extend(self.advance_stack(tuple(stack)))
            sub_rhs_offset += 1 + self.grammar_encoding[sub_rhs_offset]

        self.rule_offsets = rule_offsets
        self.stacks = stacks

    @lru_cache(maxsize=32768)
    def advance_stack(self, stack: Tuple[int]) -> List[List[int]]:
        stack = list(stack)
        if len(stack) == 0:
            return [stack]

        pos = stack[-1]

        if self.grammar_encoding[pos] > 1:
            return [stack]

        referenced_rule_id = self.grammar_encoding[pos + 1]
        rule_stride_idx = self.rule_offsets[referenced_rule_id] + 1
        stacks: List[List[int]] = []

        while self.grammar_encoding[rule_stride_idx]:
            new_stack = stack[:-1]
            if self.grammar_encoding[pos + 2]:
                new_stack.append(pos + 2)

            if self.grammar_encoding[rule_stride_idx + 1]:
                new_stack.append(rule_stride_idx + 1)

            stacks.extend(self.advance_stack(tuple(new_stack)))
            rule_stride_idx += self.grammar_encoding[rule_stride_idx] + 1

        return stacks


if __name__ == "__main__":
    # set logging level
    logging.basicConfig(level=logging.DEBUG)

    try:
        with open("examples/grammars/json.ebnf", "r") as file:
            input_text = file.read()
        state = parse_ebnf(input_text)
        print_grammar(sys.stdout, state)
        print(f"symbol_ids: \n{state.symbol_table}")

        start_rule_id = state.symbol_table["root"]
        LlamaGrammar(state.grammar_encoding, start_rule_id)

    except FileNotFoundError:
        print("Error: File 'grammar.ebnf' not found.")
    except IOError as e:
        print("Error reading file 'grammar.ebnf':", e)
