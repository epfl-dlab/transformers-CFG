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
from transformers_cfg.utf8_utils import PartialUTF8


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

    def _consume_byte_partial_match(
        self, byte: int, stacks: List[List[int]], partial_utf8: PartialUTF8
    ):
        # suppose we have code point 一, ord('一') = 19968, we need to match 3 bytes
        # we need to match 3 bytes, so we need to call _consume_byte_partial_match 3 times
        raise NotImplementedError

    def _consume_bytes_partial_match(
        self, bytes: bytes, stacks: List[List[int]], partial_utf8: PartialUTF8
    ):
        raise NotImplementedError

    def _consume_char(self, char_code_point: int, stacks: List[List[int]]):
        """
        consume a character from the stack
        char_code_point: can be a Unicode code point, including ascii code points which are in the range [0, 127]
        """
        # TODO, the below code will raise an error when the stack is empty, but why is this happening?
        # if len(stacks) == 0:
        #     raise ValueError("Stacks don't contain any stack, meaning that no character can be consumed")
        # code_point = 0 is a special case for EOS token, which should be handled by the _consume_token method
        if char_code_point == 0:
            raise ValueError("byte cannot be 0")
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
                    self.grammar_encoding[element_offset + i] <= char_code_point
                    and char_code_point <= self.grammar_encoding[element_offset + i + 1]
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

    def _accept_char(self, byte: int, stacks: List[List[int]]):
        new_stacks = self._consume_char(byte, stacks)
        return len(new_stacks) > 0

    def _consume_string(self, string: str, stacks: List[List[int]]):
        # _bytes = bytes(string, "utf-8")

        for i, char in enumerate(string):
            code_pt = ord(char)
            print(f"char: {char}, code_pt: {code_pt}")
            stacks = self._consume_char(code_pt, stacks)
            if len(stacks) > 0:
                accepted_string = string[: i + 1]
                logging.debug(f"{accepted_string} is accepted")
        return stacks

    def _accept_string(self, string: str, stacks: List[List[int]]):
        new_stacks = self._consume_string(string, stacks)
        return len(new_stacks) > 0

    def _can_stop(self, stacks: List[List[int]]):
        # This happens in practice, but maybe it shouldn't? TODO
        if len(stacks) == 0:
            return True
        # if any of the stack is empty, we can stop
        for stack in stacks:
            if len(stack) == 0:
                return True
        else:
            return False

    def _must_stop(self, stacks: List[List[int]]):
        return len(stacks) == 0 or all(len(stack) == 0 for stack in stacks)

    @lru_cache(maxsize=None)
    def char_acceptance_at_element(self, element_offset):
        """
        Caches and returns a dictionary indicating whether a Unicode character is accepted
        at a given rule position. This function considers Unicode characters, dynamically
        inserting accepted ranges into a dictionary to optimize memory usage.

        Args:
        - rule_offset: The offset in the grammar encoding where the rule starts.

        Returns:
        - A dictionary where each key is a Unicode character (or range) and the value is True if accepted.
        """
        print(f"element_offset: {element_offset}")
        acceptance = {}
        num_chars = self.grammar_encoding[element_offset]
        element_offset += 1
        for i in range(0, num_chars, 2):
            start = self.grammar_encoding[element_offset + i]
            end = self.grammar_encoding[element_offset + i + 1]
            for j in range(start, end + 1):
                acceptance[j] = True
        print(acceptance)
        return acceptance


if __name__ == "__main__":
    # set logging level

    with open("examples/grammars/debug/plus.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)
    parsed_grammar.print()
    print(f"symbol_ids: \n{parsed_grammar.symbol_table}")

    start_rule_id = parsed_grammar.symbol_table["root"]
    recognizer = GrammarRecognizer(parsed_grammar.grammar_encoding, start_rule_id)
    res = recognizer._accept_string("12222", recognizer.stacks)
    print(f"12222: {res}")
    res = recognizer._accept_string("12222+", recognizer.stacks)
    print(f"12222+: {res}")
