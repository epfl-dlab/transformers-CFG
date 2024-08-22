import logging
from functools import lru_cache
from typing import List, Tuple, Set, Optional

from transformers_cfg.parser import (
    END_OF_RULE_MARKER,
    END_OF_ALTERNATE_MARKER,
    parse_ebnf,
    REF_RULE_MARKER,
)
from transformers_cfg.utf8_utils import PartialUTF8, decode_utf8
from transformers_cfg.utils import intervals_intersect
import logging


class AcceptState:
    def __init__(self, stacks, partial_utf8):
        self.stacks = stacks
        self.partial_utf8 = partial_utf8

    @staticmethod
    def empty_state():
        return AcceptState(set(), PartialUTF8())

    def can_stop(self):
        # This happens in practice, but maybe it shouldn't? TODO
        if len(self.stacks) == 0:
            return True
        # if any of the stack is empty, we can stop
        for stack in self.stacks:
            if len(stack) == 0:
                return True
            else:
                return False

    def must_stop(self):
        return len(self.stacks) == 0 or all(len(stack) == 0 for stack in self.stacks)


class StringRecognizer:
    def __init__(
        self,
        grammar_encoding: List[int],
        start_rule_id: int = -1,
        rule_offsets: Optional[List[int]] = None,
        stacks: Optional[Set[Tuple[int]]] = None,
    ):
        # strictly speaking, we don't need to copy grammar_encoding because we don't modify it
        # but we do it anyway to be safe
        # in case where the grammar is very large, we can consider not copying it
        self.grammar_encoding = grammar_encoding
        if rule_offsets is not None:
            self.rule_offsets = rule_offsets
        else:
            if start_rule_id == -1:
                raise ValueError("start_rule_id cannot be None if rule_offsets is None")
            self.rule_offsets = self.init_rules(start_rule_id)
        # each stack is a list of indices into grammar_encoding
        # each index points to a rule's
        if stacks is not None:
            self.stacks = stacks
        else:
            if start_rule_id == -1:
                raise ValueError("start_rule_id cannot be None if stacks is None")
            self.stacks = self.init_stack(start_rule_id)
        self.start_rule_id = start_rule_id

    def init_rules(self, start_rule_id: int) -> List[int]:
        _rule_offset = 0
        rule_offsets: List = []
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

    def init_stack(self, start_rule_id: int) -> Set[Tuple[int]]:

        stacks = set()
        # Loop over alternates of start rule to build initial stacks
        sub_rhs_offset = self.rule_offsets[start_rule_id] + 1
        while self.grammar_encoding[sub_rhs_offset]:
            stack: List[int] = []
            # If alternate is nonempty, add to stack
            element_offset = sub_rhs_offset + 1
            if self.grammar_encoding[element_offset] != END_OF_ALTERNATE_MARKER:
                stack.append(element_offset)
            stacks.update(self.expand_stack_head(tuple(stack)))
            sub_rhs_offset += 1 + self.grammar_encoding[sub_rhs_offset]
        return stacks

    def get_initial_parsing_state(self) -> AcceptState:
        return AcceptState(self.init_stack(self.start_rule_id), PartialUTF8())

    def get_termination_parsing_state(self) -> AcceptState:
        return AcceptState(set(), PartialUTF8())

    @lru_cache(maxsize=32768)
    def expand_stack_head(self, stack: Tuple[int]) -> Set[Tuple[int]]:
        """
        Stack is the internal state of the recognizer(Pushdown Automaton).
        This method updates the stack by advancing it to the next element.
        If the element is a non-terminal, it expands the stack by adding the elements of the referenced rule.
        A new stack is created for each alternate of the referenced rule, so we could have multiple stacks as output.
        :param stack:
        :return:
        """
        if len(stack) == 0:
            return {stack}

        # we get the last element of the stack, which is the element we are currently processing
        cur_element_offset = stack[-1]

        # if the element is a terminal, we don't need to advance the stack
        if self.grammar_encoding[cur_element_offset] != REF_RULE_MARKER:
            return {stack}
        # the remaining case is that the element is a non-terminal, i.e. a reference to another rule
        else:
            ref_rule_id = self.grammar_encoding[cur_element_offset + 1]
            # find the offset of the referenced rule
            ref_subrule_offset = self.rule_offsets[ref_rule_id] + 1
            new_stacks: Set[Tuple[int]] = set()
            # Loop over alternates of referenced rule to build new stacks
            while self.grammar_encoding[ref_subrule_offset] != END_OF_RULE_MARKER:
                # copy the original stack without the last element
                new_stack = list(stack[:-1])
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

                new_stacks.update(self.expand_stack_head(tuple(new_stack)))
                ref_subrule_offset += self.grammar_encoding[ref_subrule_offset] + 1

            return new_stacks

    def _update_state_with_byte(
        self, byte: int, parsing_state: AcceptState
    ) -> AcceptState:
        # suppose we have code point 一, ord('一') = 19968, we need to match 3 bytes
        # we need to match 3 bytes, so we need to call _consume_byte_partial_match 3 times
        return self._update_state_with_bytes(bytes([byte]), parsing_state)

    # @lru_cache(maxsize=32768)
    def _try_accept_bytes(
        self,
        byte_seq: bytes,
        stacks: Set[Tuple[int]],
        partial_utf8: PartialUTF8,
        verbose=True,
    ):
        """
        The difference between accept_bytes and consume_bytes is that accept_bytes returns a boolean and
        consume_bytes returns a new accept state
        """
        if type(byte_seq) is list:
            byte_seq = bytes(byte_seq)
        code_points, new_partial_utf8 = decode_utf8(byte_seq, partial_utf8)
        if verbose:
            logging.debug(
                f"code_points: {code_points}; new_partial_utf8: {new_partial_utf8}"
            )
        new_stacks = self._update_state_with_code_points_for_all_stacks(
            code_points, stacks
        )

        for stack in new_stacks:

            # stack is empty, meaning that the variables are all consumed
            if len(stack) == 0:
                return True
            element_offset = stack[-1]
            if self.partial_utf8_accept_at_element(element_offset, new_partial_utf8):
                return True
        return False

    def _update_state_with_bytes(
        self,
        byte_seq: bytes,
        parsing_state: Optional[AcceptState] = None,
        verbose=True,
    ) -> AcceptState:
        if parsing_state is None:
            parsing_state = self.get_initial_parsing_state()
        stacks = parsing_state.stacks
        partial_utf8 = parsing_state.partial_utf8
        if type(byte_seq) is list:
            byte_seq = bytes(byte_seq)
        code_points, new_partial_utf8 = decode_utf8(byte_seq, partial_utf8)
        if verbose:
            logging.debug(
                f"code_points: {code_points}; new_partial_utf8: {new_partial_utf8}"
            )
        new_stacks = self._update_state_with_code_points_for_all_stacks(
            code_points, stacks
        )

        new_new_stacks = set()
        for stack in new_stacks:
            if len(stack) == 0:
                continue
            element_offset = stack[-1]
            if self.partial_utf8_accept_at_element(element_offset, new_partial_utf8):
                new_new_stacks.add(stack)
        return AcceptState(new_new_stacks, new_partial_utf8)

    ##########################
    #
    # Code point recognition
    #
    ##########################

    @lru_cache(maxsize=30000)
    def _update_state_with_code_point_for_all_stacks(
        self, code_point: int, stacks: Tuple[Tuple[int]]
    ) -> Set[Tuple[int]]:
        """
        consume a character from the stack
        char_code_point: can be a Unicode code point, including ascii code points which are in the range [0, 127]
        """
        new_stacks: Set[Tuple[int]] = set()

        if code_point == 0:
            return new_stacks
        for stack in stacks:
            new_stacks.update(
                self._update_state_with_code_point_for_single_stack(code_point, stack)
            )
        return new_stacks

    @lru_cache(maxsize=30000)
    def _update_state_with_code_point_for_single_stack(
        self, code_point: int, stack: Tuple[int]
    ) -> Set[Tuple[int]]:
        """
        consume a character from the stack
        char_code_point: can be a Unicode code point, including ascii code points which are in the range [0, 127]
        """
        # TODO, the below code will raise an error when the stack is empty, but why is this happening?
        # if len(stacks) == 0:
        #     raise ValueError("Stacks don't contain any stack, meaning that no character can be consumed")
        # code_point = 0 is a special case when the uf8 sequence is not complete, we return an empty stack
        # to indicate that the character is not accepted

        new_stacks: Set[Tuple[int]] = set()
        if code_point == 0:
            return new_stacks
        # stack is empty
        if len(stack) == 0:
            return new_stacks

        element_offset = stack[-1]

        found = self.accept_code_point_at_element(code_point, element_offset)
        if not found:
            return new_stacks

        size = self.grammar_encoding[element_offset]
        element_offset += size + 1
        new_stack = list(stack[:-1])
        if self.grammar_encoding[element_offset]:
            new_stack.append(element_offset)
        # # Explicitly convert list to tuple of int to make it hashable
        new_tuple_stack: Tuple[int, ...] = tuple(new_stack)
        return self.expand_stack_head(new_tuple_stack)

    def _update_state_with_code_points_for_all_stacks(
        self, code_points: List[int], stacks: Set[Tuple[int]], verbose=False
    ) -> Set[Tuple[int]]:
        """
        code points is a list of Unicode code points. For example, the code points for "hello" is [104, 101, 108, 108, 111]
        For unicode string "こんにちは世界", the code points are [12371, 12435, 12395, 12385, 12399, 19990, 30028]

        """
        for i, code_point in enumerate(code_points):
            # for lru_cache to work, we need to convert the list of stacks into a tuple of stacks
            tuple_stacks: Tuple[Tuple[int], ...] = tuple(stacks)
            stacks = self._update_state_with_code_point_for_all_stacks(
                code_point, tuple_stacks
            )
            if len(stacks) > 0 and verbose:
                accepted_code_point = code_points[: i + 1]
                corresponding_char = chr(code_point)
                logging.debug(
                    f"code point {accepted_code_point} corresponding to {corresponding_char} is accepted"
                )
        return stacks

    def _accept_code_points(
        self, code_points: List[int], stacks: Set[Tuple[int]], verbose=False
    ) -> bool:
        stacks = self._update_state_with_code_points_for_all_stacks(
            code_points, stacks, verbose
        )
        return len(stacks) > 0

    @lru_cache(maxsize=30000)
    def accept_code_point_at_element(
        self, code_point: int, element_offset: int
    ) -> bool:
        size = self.grammar_encoding[element_offset]
        # to make idx point to the range_start of the first range
        element_offset += 1
        for i in range(0, size, 2):
            if (
                self.grammar_encoding[element_offset + i]
                <= code_point
                <= self.grammar_encoding[element_offset + i + 1]
            ):
                return True
        return False

    #############################
    #
    # Partial UTF-8 recognition
    #
    #############################

    def partial_utf8_accept_at_element(
        self, element_offset: int, partial_utf8: PartialUTF8
    ) -> bool:
        # Extract the accumulated value and the number of remaining bytes from the partial_utf8 object.
        partial_value = partial_utf8.value
        n_remain = partial_utf8.n_remain

        # Return False if there are no remaining bytes to process or if it's an invalid UTF-8 sequence.
        if n_remain == 1 and partial_value < 2:
            return False

        # If there are no remaining bytes, this means we had already consumed a complete UTF-8 sequence.
        if n_remain <= 0:
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

    #############################
    #
    # String recognition
    #
    #############################

    def _update_state_with_string(self, string: str, parsing_state: AcceptState):
        # _bytes = bytes(string, "utf-8")
        code_points = [ord(char) for char in string]
        stacks = self._update_state_with_code_points_for_all_stacks(
            code_points, parsing_state.stacks
        )
        return AcceptState(stacks, parsing_state.partial_utf8)

    def _accept_prefix(self, string: str, parsing_state: Optional[AcceptState] = None):
        if parsing_state is None:
            parsing_state = self.get_initial_parsing_state()
        new_parsing_state = self._update_state_with_string(string, parsing_state)
        return len(new_parsing_state.stacks) > 0

    def _accept_string(self, string: str, parsing_state: Optional[AcceptState] = None):
        if parsing_state is None:
            parsing_state = self.get_initial_parsing_state()
        new_parsing_state = self._update_state_with_string(string, parsing_state)
        at_least_one_stack_is_empty = any(
            len(stack) == 0 for stack in new_parsing_state.stacks
        )
        return at_least_one_stack_is_empty

    #############################
    #
    # Not Used
    #
    #############################

    # For each sub-rule in the grammar, cache whether each byte is accepted.
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


# backward compatibility, add alias of StringRecognizer to GrammarRecognizer
GrammarRecognizer = StringRecognizer

if __name__ == "__main__":
    # set logging level

    with open("examples/grammars/japanese.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)
    logging.debug(f"symbol_ids: \n{parsed_grammar.symbol_table}")

    start_rule_id = parsed_grammar.symbol_table["root"]
    recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

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

    parsing_state = AcceptState(recognizer.stacks, PartialUTF8())
    for i, byte in enumerate(byte_tokens):
        parsing_state = recognizer._update_state_with_bytes(byte, parsing_state)
        logging.debug(f"new partial utf8: {parsing_state.partial_utf8}")
        if len(parsing_state.stacks) > 0:
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
