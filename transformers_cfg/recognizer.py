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
from transformers_cfg.utf8_utils import PartialUTF8, decode_utf8
from transformers_cfg.utils import intervals_intersect
import logging

logging.basicConfig(level=logging.DEBUG)


class GrammarRecognizer:
    def __init__(
        self,
        grammar_encoding: List[int],
        start_rule_id: int = None,
        rule_offsets: List[int] = None,
        stacks: List[List[int]] = None,
        partial_utf8: PartialUTF8 = None,
    ):
        # strictly speaking, we don't need to copy grammar_encoding because we don't modify it
        # but we do it anyway to be safe
        # in case where the grammar is very large, we can consider not copying it
        self.grammar_encoding = grammar_encoding
        if rule_offsets is not None:
            self.rule_offsets = rule_offsets
        else:
            if start_rule_id is None:
                raise ValueError("start_rule_id cannot be None if rule_offsets is None")
            self.rule_offsets = self.init_rules(start_rule_id)
        # each stack is a list of indices into grammar_encoding
        # each index points to a rule's
        if stacks is not None:
            self.stacks = stacks
        else:
            if start_rule_id is None:
                raise ValueError("start_rule_id cannot be None if stacks is None")
            self.stacks: List[List[int]] = self.init_stack(start_rule_id)
        self.partial_utf8 = partial_utf8

    def init_rules(self, start_rule_id: int) -> List[int]:
        _rule_offset = 0
        rule_offsets = []
        # Build `rules` as an array of rule IDs to their positions in `grammar_src`
        while self.grammar_encoding[_rule_offset] != 0xFFFF:
            rule_id = self.grammar_encoding[_rule_offset]
            # store the offset idx
            if len(rule_offsets) <= rule_id:
                rule_offsets.extend([-1] * (rule_id - len(rule_offsets) + 1))
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

    def _consume_byte_partial_match(self, byte: int, stacks: List[List[int]]):
        # suppose we have code point 一, ord('一') = 19968, we need to match 3 bytes
        # we need to match 3 bytes, so we need to call _consume_byte_partial_match 3 times
        raise NotImplementedError

    def _consume_bytes_partial_match(
        self, bytes: bytes, stacks: List[List[int]], partial_utf8: PartialUTF8
    ):
        code_points, new_partial_utf8 = decode_utf8(bytes, partial_utf8)
        logging.debug(
            f"code_points: {code_points}; new_partial_utf8: {new_partial_utf8}"
        )
        new_stacks = self._consume_code_points(code_points, stacks)
        # pdb.set_trace()

        new_new_stacks = []
        for stack in new_stacks:
            if len(stack) == 0:
                continue
            element_offset = stack[-1]
            if self.valid_partial_match(element_offset, new_partial_utf8):
                new_new_stacks.append(stack)
        return new_new_stacks, new_partial_utf8

    def _consume_char_code_point(self, char_code_point: int, stacks: List[List[int]]):
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

            found = self.valid_char_code_point_match(char_code_point, element_offset)
            if not found:
                continue

            size = self.grammar_encoding[element_offset]
            element_offset += size + 1
            new_stack = stack[:-1]
            if self.grammar_encoding[element_offset]:
                new_stack.append(element_offset)
            new_stacks.extend(self.advance_stack(tuple(new_stack)))
        return new_stacks

    def valid_char_code_point_match(
        self, char_code_point: int, element_offset: int
    ) -> bool:
        size = self.grammar_encoding[element_offset]
        # to make idx point to the range_start of the first range
        element_offset += 1
        for i in range(0, size, 2):
            if (
                self.grammar_encoding[element_offset + i]
                <= char_code_point
                <= self.grammar_encoding[element_offset + i + 1]
            ):
                return True
        return False

    def valid_partial_match(
        self, element_offset: int, partial_utf8: PartialUTF8
    ) -> bool:
        # Extract the accumulated value and the number of remaining bytes from the partial_utf8 object.
        partial_value = partial_utf8.value
        n_remain = partial_utf8.n_remain

        # Return False if there are no remaining bytes to process or if it's an invalid UTF-8 sequence.
        if n_remain < 0 or (n_remain == 1 and partial_value < 2):
            return False

        # If there are no remaining bytes, this means we had already consumed a complete UTF-8 sequence.
        if n_remain == 0:
            return True

        # Calculate the lowest possible Unicode code point that can be formed with the remaining bytes.
        low = partial_value << (n_remain * 6)
        # Calculate the highest possible Unicode code point by setting all remaining bits to 1.
        high = low | ((1 << (n_remain * 6)) - 1)

        # If the low end of the range is 0 and a specific number of bytes remain, adjust low to the minimum value
        # that can be represented with that number of bytes. This accounts for UTF-8 encoding rules.
        if low == 0:
            if n_remain == 2:
                low = 1 << 11  # Minimum value representable with 2 additional bytes.
            elif n_remain == 3:
                low = 1 << 16  # Minimum value representable with 3 additional bytes.

        # Get the size of the grammar rule starting at the current element_offset.
        size = self.grammar_encoding[element_offset]
        # Move the element_offset to the start of the grammar rule's definition.
        element_offset += 1

        # Iterate over the grammar rule, checking if the range defined by low-high overlaps with any specified ranges.
        for i in range(0, size, 2):
            # If the current range (specified in the grammar encoding) overlaps with the low-high range, return True.
            if intervals_intersect(
                low,
                high,
                self.grammar_encoding[element_offset + i],
                self.grammar_encoding[element_offset + i + 1],
            ):
                return True

        # If no overlap is found with any of the ranges, return False, indicating no valid partial match.
        return False

    def _accept_char_code_point(self, code_point: int, stacks: List[List[int]]):
        new_stacks = self._consume_char_code_point(code_point, stacks)
        return len(new_stacks) > 0

    def _consume_string(self, string: str, stacks: List[List[int]]):
        # _bytes = bytes(string, "utf-8")
        code_points = [ord(char) for char in string]
        stacks = self._consume_code_points(code_points, stacks)
        return stacks

    def _consume_code_points(self, code_points: List[int], stacks: List[List[int]]):
        for i, code_point in enumerate(code_points):
            stacks = self._consume_char_code_point(code_point, stacks)
            if len(stacks) > 0:
                accepted_code_point = code_points[: i + 1]
                corresponding_char = chr(code_point)
                logging.debug(
                    f"code point {accepted_code_point} corresponding to {corresponding_char} is accepted"
                )
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
        logging.debug(f"element_offset: {element_offset}")
        acceptance = {}
        num_chars = self.grammar_encoding[element_offset]
        element_offset += 1
        for i in range(0, num_chars, 2):
            start = self.grammar_encoding[element_offset + i]
            end = self.grammar_encoding[element_offset + i + 1]
            for j in range(start, end + 1):
                acceptance[j] = True
        logging.debug(acceptance)
        return acceptance


if __name__ == "__main__":
    # set logging level

    with open("examples/grammars/japanese.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)
    parsed_grammar.logging.debug()
    logging.debug(f"symbol_ids: \n{parsed_grammar.symbol_table}")

    start_rule_id = parsed_grammar.symbol_table["root"]
    recognizer = GrammarRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

    japanese = "こんにちは世界"  # "こんにちは世界" doesn't work
    bytes_japanese = bytes(japanese, "utf-8")
    logging.debug(f"bytes_japanese: {bytes_japanese} of length {len(bytes_japanese)}")
    # こんにちは世界

    partial_utf8 = PartialUTF8()

    #######################
    # Japanese
    #######################

    # logging.debug(f"bytes_japanese: {bytes_japanese} of length {len(bytes_japanese)}")
    # head_bytes = bytes_japanese[:8]
    # # partial_utf8 = PartialUTF8()
    # new_stacks, new_partial_utf8 = recognizer._consume_bytes_partial_match(head_bytes, recognizer.stacks, partial_utf8)
    # if len(new_stacks) > 0:
    #     logging.debug("japanese is accepted")
    # else:
    #     logging.debug("japanese is not accepted")

    #######################
    # Now consider the case of progressive matching
    #######################

    byte_tokens = [bytes_japanese[i] for i in range(len(bytes_japanese))]
    # cast into bytes
    byte_tokens = [bytes([byte]) for byte in byte_tokens]

    new_partial_utf8 = PartialUTF8()
    new_stacks = recognizer.stacks
    for i, byte in enumerate(byte_tokens):
        new_stacks, new_partial_utf8 = recognizer._consume_bytes_partial_match(
            byte, new_stacks, new_partial_utf8
        )
        logging.debug(f"new partial utf8: {new_partial_utf8}")
        if len(new_stacks) > 0:
            logging.debug(f"byte {byte} is accepted")
        else:
            logging.debug(f"byte {byte} is not accepted")

    # korean = "안녕하세요"
    # korean_bytes = bytes(korean, "utf-8")
    # head_bytes = korean_bytes[:8]
    # logging.debug(f"korean_bytes: {korean_bytes} of length {len(korean_bytes)}")
    # new_stacks, new_partial_utf8 = recognizer._consume_bytes_partial_match(head_bytes, recognizer.stacks, partial_utf8)
    # if len(new_stacks) > 0:
    #     logging.debug("korean is accepted")
    # else:
    #     logging.debug("korean is not accepted")

    # res = recognizer._accept_string("12222", recognizer.stacks)
    # logging.debug(f"12222: {res}")
    # res = recognizer._accept_string("12222+", recognizer.stacks)
    # logging.debug(f"12222+: {res}")
