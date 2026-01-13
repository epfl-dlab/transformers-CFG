"""
String recognition for grammar-constrained parsing.

This module provides the core string recognition functionality using a pushdown
automaton approach. It orchestrates grammar state management, stack operations,
and character matching to validate strings against EBNF grammars.

Key Classes:
    AcceptState: Represents parsing state with stacks and partial UTF-8 data
    StringRecognizer: Main orchestrator for grammar-based string recognition

Architecture:
    StringRecognizer composes three focused components:
    - GrammarState: Manages grammar encoding and rule lookups
    - StackManager: Handles parsing stack operations and expansion
    - CharacterMatcher: Matches characters against grammar rules

Example:
    >>> from transformers_cfg.parser import parse_ebnf
    >>> from transformers_cfg.recognizer import StringRecognizer
    >>>
    >>> # Parse an EBNF grammar
    >>> parsed = parse_ebnf('root ::= "hello" | "world"')
    >>>
    >>> # Create recognizer
    >>> recognizer = StringRecognizer(
    ...     parsed.grammar_encoding,
    ...     start_rule_id=parsed.symbol_table["root"]
    ... )
    >>>
    >>> # Check if string is accepted
    >>> state = recognizer.get_initial_parsing_state()
    >>> state = recognizer._update_state_with_string("hello", state)
    >>> accepted = len(state.stacks) > 0  # True if accepted

See Also:
    - transformers_cfg.grammar: Grammar state management components
    - transformers_cfg.parser: EBNF grammar parsing
    - transformers_cfg.token_grammar_recognizer: Token-level recognition
"""

import logging
from functools import lru_cache
from typing import List, Tuple, Set, Optional

from transformers_cfg.parser import (
    parse_ebnf,
)
from transformers_cfg.utf8_utils import PartialUTF8, decode_utf8
from transformers_cfg.grammar import GrammarState, StackManager, CharacterMatcher
from transformers_cfg._compat import deprecated_method


class AcceptState:
    def __init__(self, stacks: Set[Tuple[int]], partial_utf8: PartialUTF8):
        self.stacks = stacks
        self.partial_utf8 = partial_utf8

    @staticmethod
    def empty_state() -> "AcceptState":
        return AcceptState(set(), PartialUTF8())

    def can_stop(self) -> bool:
        # This happens in practice, but maybe it shouldn't? TODO
        if len(self.stacks) == 0:
            return True
        # if any of the stack is empty, we can stop
        for stack in self.stacks:
            if len(stack) == 0:
                return True
            else:
                return False

    def must_stop(self) -> bool:
        return len(self.stacks) == 0 or all(len(stack) == 0 for stack in self.stacks)


class StringRecognizer:
    """
    Orchestrates grammar-based string recognition.

    StringRecognizer coordinates GrammarState, StackManager, and CharacterMatcher
    to implement a pushdown automaton for grammar-constrained string recognition.

    This class maintains backward compatibility with the original API while
    internally delegating to focused component classes.

    Attributes:
        grammar_encoding: Flattened integer array encoding the grammar (for compatibility)
        grammar_state: Manages grammar encoding and rule lookups
        stack_manager: Handles parsing stack operations
        character_matcher: Matches characters against grammar rules
        stacks: Current set of parsing stacks
        start_rule_id: ID of the start rule

    Example:
        >>> recognizer = StringRecognizer(grammar_encoding, start_rule_id=0)
        >>> state = recognizer.get_initial_parsing_state()
        >>> state = recognizer._update_state_with_bytes(b"hello", state)
    """

    def __init__(
        self,
        grammar_encoding: List[int],
        start_rule_id: int = -1,
        rule_offsets: Optional[List[int]] = None,
        stacks: Optional[Set[Tuple[int]]] = None,
    ):
        """
        Initialize string recognizer with grammar encoding.

        Args:
            grammar_encoding: Flattened integer array encoding the grammar
            start_rule_id: ID of the grammar's start rule
            rule_offsets: Pre-computed rule offset table (optional, computed if None)
            stacks: Pre-initialized stacks (optional, computed if None)

        Raises:
            ValueError: If start_rule_id is invalid or rule_offsets/stacks are None
        """
        # Store grammar encoding for backward compatibility
        self.grammar_encoding = grammar_encoding
        self.start_rule_id = start_rule_id

        # Create composed components
        self._grammar_state = GrammarState(grammar_encoding, start_rule_id)
        self._stack_manager = StackManager(self._grammar_state)
        self._character_matcher = CharacterMatcher(self._grammar_state)

        # Initialize rule offsets (use provided or compute from grammar state)
        if rule_offsets is not None:
            self.rule_offsets = rule_offsets
        else:
            if start_rule_id == -1:
                raise ValueError("start_rule_id cannot be -1 if rule_offsets is None")
            self.rule_offsets = self._grammar_state.rule_offsets

        # Initialize stacks (use provided or compute from stack manager)
        if stacks is not None:
            self.stacks = stacks
        else:
            if start_rule_id == -1:
                raise ValueError("start_rule_id cannot be -1 if stacks is None")
            self.stacks = self._stack_manager.create_initial_stacks()

    # Properties for accessing composed components
    @property
    def grammar_state(self) -> GrammarState:
        """Access to grammar state component."""
        return self._grammar_state

    @property
    def stack_manager(self) -> StackManager:
        """Access to stack manager component."""
        return self._stack_manager

    @property
    def character_matcher(self) -> CharacterMatcher:
        """Access to character matcher component."""
        return self._character_matcher

    def init_rules(self, start_rule_id: int) -> List[int]:
        """
        Build rule offset table (backward compatibility wrapper).

        Deprecated: New code should use grammar_state.rule_offsets directly.
        """
        return self._grammar_state.rule_offsets

    def init_stack(self, start_rule_id: int) -> Set[Tuple[int]]:
        """
        Create initial parsing stacks (backward compatibility wrapper).

        Deprecated: New code should use stack_manager.create_initial_stacks() directly.
        """
        return self._stack_manager.create_initial_stacks()

    def get_initial_parsing_state(self) -> AcceptState:
        return AcceptState(self.init_stack(self.start_rule_id), PartialUTF8())

    def get_termination_parsing_state(self) -> AcceptState:
        return AcceptState(set(), PartialUTF8())

    def expand_stack_head(self, stack: Tuple[int]) -> Set[Tuple[int]]:
        """
        Expand stack by processing rule references at the top (backward compatibility wrapper).

        Deprecated: New code should use stack_manager.expand_stack_head() directly.

        Args:
            stack: Current parsing stack (tuple of offsets into grammar encoding)

        Returns:
            Set of stacks after expansion.
        """
        return self._stack_manager.expand_stack_head(stack)

    def _update_state_with_byte(
        self, byte: int, parsing_state: AcceptState
    ) -> AcceptState:
        # suppose we have code point 一, ord('一') = 19968, we need to match 3 bytes
        # we need to match 3 bytes, so we need to call _consume_byte_partial_match 3 times
        return self._update_state_with_bytes(bytes([byte]), parsing_state)

    # @lru_cache(maxsize=32768)
    def _can_accept_bytes(
        self,
        byte_seq: bytes,
        stacks: Set[Tuple[int]],
        partial_utf8: PartialUTF8,
        verbose=True,
    ):
        """
        Check if byte sequence can be accepted by any of the parsing stacks.

        Tests whether the given byte sequence is valid according to the grammar
        rules for at least one of the provided parsing stacks.

        Args:
            byte_seq: Byte sequence to test
            stacks: Set of parsing stacks to test against
            partial_utf8: Partial UTF-8 state from previous processing
            verbose: Enable debug logging

        Returns:
            True if byte sequence can be accepted, False otherwise

        Note:
            This method returns a boolean indicating acceptance without modifying state.
            Use _update_state_with_bytes() to actually consume bytes and update state.
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
            # stack is empty, meaning that all elements are consumed
            if len(stack) == 0:
                return True
            element_offset = stack[-1]
            if self.partial_utf8_accept_at_element(element_offset, new_partial_utf8):
                return True
        return False

    @deprecated_method("_can_accept_bytes")
    def _try_accept_bytes(
        self,
        byte_seq: bytes,
        stacks: Set[Tuple[int]],
        partial_utf8: PartialUTF8,
        verbose=True,
    ):
        """Deprecated: Use _can_accept_bytes() instead."""
        return self._can_accept_bytes(byte_seq, stacks, partial_utf8, verbose)

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
                new_new_stacks.add(stack)
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
        """
        Check if element accepts code point (backward compatibility wrapper).

        Delegates to character_matcher.accepts_code_point().
        """
        return self._character_matcher.accepts_code_point(code_point, element_offset)

    #############################
    #
    # Partial UTF-8 recognition
    #
    #############################

    def partial_utf8_accept_at_element(
        self, element_offset: int, partial_utf8: PartialUTF8
    ) -> bool:
        """
        Check if element can accept partial UTF-8 sequence (backward compatibility wrapper).

        Delegates to character_matcher.accepts_partial_utf8().
        """
        return self._character_matcher.accepts_partial_utf8(
            element_offset, partial_utf8
        )

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
