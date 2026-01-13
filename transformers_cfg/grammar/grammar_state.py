"""Grammar encoding state management."""

import logging
from typing import List

from transformers_cfg.parser import END_OF_RULE_MARKER, END_OF_GRAMMAR_MARKER

logger = logging.getLogger(__name__)


class GrammarState:
    """
    Manages grammar encoding and provides efficient rule lookups.

    The GrammarState stores the internal representation of a parsed grammar
    and builds an index for O(1) rule offset lookups. It acts as a read-only
    view over the grammar encoding.

    Attributes:
        grammar_encoding: Flattened integer array encoding the grammar rules
        start_rule_id: ID of the grammar's start rule
        rule_offsets: Mapping from rule ID to position in grammar_encoding

    Example:
        >>> from transformers_cfg.parser import parse_ebnf
        >>> parsed = parse_ebnf('root ::= "hello"')
        >>> state = GrammarState(parsed.grammar_encoding, start_rule_id=0)
        >>> offset = state.get_rule_offset(0)
    """

    def __init__(self, grammar_encoding: List[int], start_rule_id: int):
        """
        Initialize grammar state from encoded grammar.

        Args:
            grammar_encoding: Flattened integer array encoding the grammar
            start_rule_id: ID of the start rule for parsing

        Raises:
            ValueError: If grammar encoding is invalid or start_rule_id not found
        """
        self.grammar_encoding = grammar_encoding
        self.start_rule_id = start_rule_id
        self.rule_offsets = self._build_rule_offsets()

        # Validate that start rule exists
        if (
            start_rule_id >= len(self.rule_offsets)
            or self.rule_offsets[start_rule_id] == -1
        ):
            raise ValueError(f"Start rule ID {start_rule_id} not found in grammar")

    def _build_rule_offsets(self) -> List[int]:
        """
        Build index from rule ID to position in grammar encoding.

        Returns:
            List where rule_offsets[rule_id] = offset in grammar_encoding
        """
        rule_offset = 0
        rule_offsets: List[int] = []

        # Build array mapping rule IDs to their positions in grammar_encoding
        while self.grammar_encoding[rule_offset] != END_OF_GRAMMAR_MARKER:
            rule_id = self.grammar_encoding[rule_offset]

            # Extend array if needed to accommodate this rule ID
            if len(rule_offsets) <= rule_id:
                rule_offsets.extend([-1] * (rule_id - len(rule_offsets) + 1))

            rule_offsets[rule_id] = rule_offset

            # Skip to next rule by traversing this rule's alternatives
            simple_rhs_offset = rule_offset + 1

            # Skip rule alternates
            while self.grammar_encoding[simple_rhs_offset] != END_OF_RULE_MARKER:
                # Each alternate starts with its length, skip by length + 1
                simple_rhs_offset = (
                    simple_rhs_offset + 1 + self.grammar_encoding[simple_rhs_offset]
                )

            # Skip the END_OF_RULE_MARKER
            rule_offset = simple_rhs_offset + 1

        # Validate start rule exists in encoding
        if self.start_rule_id < len(rule_offsets):
            retrieved_start_rule_id = self.grammar_encoding[
                rule_offsets[self.start_rule_id]
            ]
            assert retrieved_start_rule_id == self.start_rule_id

        return rule_offsets

    def get_rule_offset(self, rule_id: int) -> int:
        """
        Get offset of a rule in the grammar encoding.

        Args:
            rule_id: ID of the rule to look up

        Returns:
            Offset into grammar_encoding where the rule starts

        Raises:
            IndexError: If rule_id is out of bounds
            ValueError: If rule_id is not defined in grammar
        """
        if rule_id >= len(self.rule_offsets):
            raise IndexError(f"Rule ID {rule_id} is out of bounds")

        offset = self.rule_offsets[rule_id]
        if offset == -1:
            raise ValueError(f"Rule ID {rule_id} is not defined in grammar")

        return offset

    def get_element_size(self, element_offset: int) -> int:
        """
        Get size of element at given offset.

        Args:
            element_offset: Offset into grammar encoding

        Returns:
            Size of the element (used for skipping)
        """
        return self.grammar_encoding[element_offset]
