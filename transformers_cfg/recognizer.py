import copy
import logging
from functools import lru_cache
from typing import List, Tuple, Dict

from transformers_cfg.parser import (
    END_OF_RULE_MARKER,
    END_OF_ALTERNATE_MARKER,
    parse_ebnf,
    REF_RULE_MARKER,
)


class GrammarRecognizer:
    def __init__(self, grammar_encoding: List[int], start_rule_id: int):
        # strictly speaking, we don't need to copy grammar_encoding because we don't modify it
        # but we do it anyway to be safe
        # in case where the grammar is very large, we can consider not copying it
        self.grammar_encoding = copy.deepcopy(grammar_encoding)
        self.rule_offsets: Dict[int, int] = self.init_rules(start_rule_id)
        # each stack is a list of indices into grammar_encoding
        # each index points to a rule's
        self.stacks: List[List[int]] = self.init_stack(start_rule_id)

    def init_rules(self, start_rule_id: int) -> Dict[int, int]:
        _rule_offset = 0
        rule_offsets = {}
        # Build `rules` as a dictionary of rule IDs to their positions in `grammar_src`
        while self.grammar_encoding[_rule_offset] != 0xFFFF:
            rule_id = self.grammar_encoding[_rule_offset]
            # store the offset idx
            rule_offsets[rule_id] = _rule_offset

            # Skip rule ID
            # _rule_offset += 1
            simple_rhs_offset = _rule_offset + 1

            # Skip rule alternates
            while self.grammar_encoding[simple_rhs_offset] != END_OF_RULE_MARKER:
                simple_rhs_offset = (
                    simple_rhs_offset + 1 + self.grammar_encoding[simple_rhs_offset]
                )

            # Skip 0 denoting end of rule
            # _rule_offset += 1
            _rule_offset = simple_rhs_offset + 1

        retrieved_start_rule_id = self.grammar_encoding[rule_offsets[start_rule_id]]
        assert retrieved_start_rule_id == start_rule_id

        return rule_offsets

    def init_stack(self, start_rule_id: int) -> List[List[int]]:

        stacks = []
        # Loop over alternates of start rule to build initial stacks
        sub_rhs_offset = self.rule_offsets[start_rule_id] + 1
        while self.grammar_encoding[sub_rhs_offset]:
            stack: List[int] = []
            # If alternate is nonempty, add to stack
            element_offset = sub_rhs_offset + 1
            if self.grammar_encoding[element_offset] != END_OF_ALTERNATE_MARKER:
                stack.append(element_offset)
            stacks.extend(self.advance_stack(tuple(stack)))
            sub_rhs_offset += 1 + self.grammar_encoding[sub_rhs_offset]
        return stacks

    @lru_cache(maxsize=32768)
    def advance_stack(self, stack: Tuple[int]) -> List[List[int]]:
        stack = list(stack)
        if len(stack) == 0:
            return [stack]

        # we get the last element of the stack, which is the element we are currently processing
        cur_element_offset = stack[-1]

        # if the element is a terminal, we don't need to advance the stack
        if self.grammar_encoding[cur_element_offset] != REF_RULE_MARKER:
            return [stack]
        # the remaining case is that the element is a non-terminal, i.e. a reference to another rule
        else:
            ref_rule_id = self.grammar_encoding[cur_element_offset + 1]
            # find the offset of the referenced rule
            ref_subrule_offset = self.rule_offsets[ref_rule_id] + 1
            new_stacks: List[List[int]] = []
            # Loop over alternates of referenced rule to build new stacks
            while self.grammar_encoding[ref_subrule_offset] != END_OF_RULE_MARKER:
                # copy the original stack without the last element
                new_stack = stack[:-1]
                # if the rule ref is followed by another element, we add it to the stack
                next_element_offset = cur_element_offset + 2
                if (
                    self.grammar_encoding[next_element_offset]
                    != END_OF_ALTERNATE_MARKER
                ):
                    new_stack.append(next_element_offset)

                # if the referenced rule is not empty, we add its element offset to the stack
                ref_element_offset = ref_subrule_offset + 1
                if self.grammar_encoding[ref_element_offset] != END_OF_ALTERNATE_MARKER:
                    new_stack.append(ref_element_offset)

                new_stacks.extend(self.advance_stack(tuple(new_stack)))
                ref_subrule_offset += self.grammar_encoding[ref_subrule_offset] + 1

            return new_stacks

    def _consume_char(self, byte, stacks):
        new_stacks = []
        for stack in stacks:
            # stack is empty
            if not stack:
                continue

            element_offset = stack[-1]
            size = self.grammar_encoding[element_offset]

            # to make idx point to the range_start of the first range
            element_offset += 1
            found = False
            for i in range(0, size, 2):
                if (
                    self.grammar_encoding[element_offset + i] <= byte
                    and byte <= self.grammar_encoding[element_offset + i + 1]
                ):
                    found = True
                    break
            if not found:
                continue

            element_offset += size
            new_stack = stack[:-1]
            if self.grammar_encoding[element_offset]:
                new_stack.append(element_offset)
            new_stacks.extend(self.advance_stack(tuple(new_stack)))

        return new_stacks

    def _consume_string(self, string: str, stacks: List[List[int]]):
        _bytes = bytes(string, "utf-8")
        for byte in _bytes:
            stacks = self._consume_char(byte, stacks)
        return stacks


if __name__ == "__main__":
    # set logging level
    logging.basicConfig(level=logging.DEBUG)

    try:
        with open("examples/grammars/debug/plus.ebnf", "r") as file:
            input_text = file.read()
        parsed_grammar = parse_ebnf(input_text)
        parsed_grammar.print()
        print(f"symbol_ids: \n{parsed_grammar.symbol_table}")

        start_rule_id = parsed_grammar.symbol_table["root"]
        json_automaton = GrammarRecognizer(
            parsed_grammar.grammar_encoding, start_rule_id
        )
        print(f"rule_offsets: \n{json_automaton.rule_offsets}")
        print(f"stacks: \n{json_automaton.stacks}")
        out_stacks = json_automaton._consume_char(ord("1"), json_automaton.stacks)
        print(f"out_stacks: \n{out_stacks}")
    except FileNotFoundError:
        print("Error: File 'grammar.ebnf' not found.")
    except IOError as e:
        print("Error reading file 'grammar.ebnf':", e)
