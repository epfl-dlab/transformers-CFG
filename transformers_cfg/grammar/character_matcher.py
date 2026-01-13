"""Character and code point matching against grammar rules."""

import logging
from functools import lru_cache

from transformers_cfg.utf8_utils import PartialUTF8
from transformers_cfg.utils import intervals_intersect
from .grammar_state import GrammarState

logger = logging.getLogger(__name__)


class CharacterMatcher:
    """
    Matches characters/code points against grammar rules.

    This component tests whether characters (represented as Unicode code points)
    are accepted by grammar elements. It handles both complete code points and
    partial UTF-8 sequences.

    Attributes:
        grammar: GrammarState containing the grammar encoding

    Note:
        Matching results are cached via @lru_cache for performance.

    Example:
        >>> matcher = CharacterMatcher(grammar_state)
        >>> element_offset = 42  # Some element in grammar
        >>> if matcher.accepts_code_point(ord('a'), element_offset):
        ...     print("Character 'a' is accepted")
    """

    def __init__(self, grammar: GrammarState):
        """
        Initialize character matcher with grammar state.

        Args:
            grammar: GrammarState to use for matching operations
        """
        self.grammar = grammar

    @lru_cache(maxsize=30000)
    def accepts_code_point(self, code_point: int, element_offset: int) -> bool:
        """
        Check if a grammar element accepts a given code point.

        Tests whether the code point falls within any of the character ranges
        specified by the grammar element.

        Args:
            code_point: Unicode code point to test (e.g., ord('a') = 97)
            element_offset: Offset of grammar element to test against

        Returns:
            True if the code point is accepted by this element, False otherwise

        Example:
            >>> matcher.accepts_code_point(ord('a'), element_offset)
            True
        """
        size = self.grammar.grammar_encoding[element_offset]

        # Move to first range start (skip size marker)
        range_offset = element_offset + 1

        # Check if code point falls within any of the element's ranges
        # Ranges are stored as pairs: [start1, end1, start2, end2, ...]
        for i in range(0, size, 2):
            range_start = self.grammar.grammar_encoding[range_offset + i]
            range_end = self.grammar.grammar_encoding[range_offset + i + 1]

            if range_start <= code_point <= range_end:
                return True

        return False

    def accepts_partial_utf8(
        self, element_offset: int, partial_utf8: PartialUTF8
    ) -> bool:
        """
        Check if element can accept a partial UTF-8 sequence.

        When processing bytes incrementally, we may have an incomplete UTF-8
        multi-byte sequence. This method checks if completing that sequence
        could result in a code point accepted by the grammar element.

        Args:
            element_offset: Offset of grammar element to test against
            partial_utf8: Incomplete UTF-8 sequence state

        Returns:
            True if any completion of the partial sequence would be accepted

        Note:
            This is used during byte-by-byte parsing to determine if we should
            continue accepting bytes to complete a multi-byte UTF-8 character.
        """
        partial_value = partial_utf8.value
        n_remain = partial_utf8.n_remain

        # Return False for invalid UTF-8 sequences
        if n_remain == 1 and partial_value < 2:
            return False

        # If no remaining bytes, we already have a complete sequence
        if n_remain <= 0:
            return True

        # Calculate range of possible code points when sequence is completed
        # Low: all remaining continuation bytes are 0x00
        low = partial_value << (n_remain * 6)

        # High: all remaining continuation bytes are 0x3F (maximum)
        high = low | ((1 << (n_remain * 6)) - 1)

        # Adjust low for UTF-8 encoding rules (minimum values for byte counts)
        if low == 0:
            if n_remain == 2:
                low = 1 << 11  # Minimum 3-byte UTF-8 value
            elif n_remain == 3:
                low = 1 << 16  # Minimum 4-byte UTF-8 value

        # Get element size and move to first range
        size = self.grammar.grammar_encoding[element_offset]
        range_offset = element_offset + 1

        # Check if possible code point range overlaps with any grammar ranges
        for i in range(0, size, 2):
            grammar_range_start = self.grammar.grammar_encoding[range_offset + i]
            grammar_range_end = self.grammar.grammar_encoding[range_offset + i + 1]

            if intervals_intersect(low, high, grammar_range_start, grammar_range_end):
                return True

        return False
