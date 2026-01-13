"""Parsing stack management for grammar recognition."""

import logging
from functools import lru_cache
from typing import Set, Tuple

from transformers_cfg.parser import (
    END_OF_ALTERNATE_MARKER,
    END_OF_RULE_MARKER,
    REF_RULE_MARKER,
)
from .grammar_state import GrammarState

logger = logging.getLogger(__name__)


class StackManager:
    """
    Manages parsing stacks for pushdown automaton-based grammar recognition.

    Grammar recognition uses a set of parsing stacks (one per parse path).
    Each stack contains offsets into the grammar encoding representing the
    current parse state. This manager handles stack creation, expansion,
    and rule reference resolution.

    Attributes:
        grammar: GrammarState containing the grammar encoding

    Note:
        Stack expansion is cached via @lru_cache for performance. Each unique
        stack configuration is expanded at most once.

    Example:
        >>> manager = StackManager(grammar_state)
        >>> initial = manager.create_initial_stacks()
        >>> expanded = manager.expand_stack_head(list(initial)[0])
    """

    def __init__(self, grammar: GrammarState):
        """
        Initialize stack manager with grammar state.

        Args:
            grammar: GrammarState to use for stack operations
        """
        self.grammar = grammar

    def create_initial_stacks(self) -> Set[Tuple[int]]:
        """
        Create initial stacks for the start rule.

        Returns:
            Set containing initial parsing stacks for the grammar's start rule
        """
        stacks = set()

        # Get offset of start rule
        start_rule_offset = self.grammar.get_rule_offset(self.grammar.start_rule_id)

        # Loop over all alternatives of start rule to build initial stacks
        sub_rhs_offset = start_rule_offset + 1
        while self.grammar.grammar_encoding[sub_rhs_offset] != END_OF_RULE_MARKER:
            # Get first element of this alternative (skip the length marker)
            element_offset = sub_rhs_offset + 1

            # If alternative is non-empty, create stack and expand it
            if self.grammar.grammar_encoding[element_offset] != END_OF_ALTERNATE_MARKER:
                initial_stack = (element_offset,)
                stacks.update(self.expand_stack_head(initial_stack))

            # Move to next alternative (skip over current alternative's length)
            sub_rhs_offset += 1 + self.grammar.grammar_encoding[sub_rhs_offset]

        return stacks

    def _is_rule_reference(self, element_offset: int) -> bool:
        """
        Check if the grammar element at given offset is a non-terminal rule reference.

        Args:
            element_offset: Offset into grammar encoding

        Returns:
            True if element is a rule reference, False if it's a terminal
        """
        return self.grammar.grammar_encoding[element_offset] == REF_RULE_MARKER

    def _get_referenced_rule_id(self, element_offset: int) -> int:
        """
        Get the rule ID that a reference element points to.

        Args:
            element_offset: Offset of the REF_RULE_MARKER element

        Returns:
            Rule ID being referenced
        """
        return self.grammar.grammar_encoding[element_offset + 1]

    def _create_alternative_stack(
        self,
        original_stack: Tuple[int],
        current_element_offset: int,
        alternative_offset: int,
    ) -> Tuple[int]:
        """
        Create a new stack for one alternative of a rule reference.

        When expanding a rule reference, we create a new stack that includes:
        1. All elements from the original stack except the rule reference
        2. The continuation element (if any) after the rule reference
        3. The first element of the alternative (if non-empty)

        Args:
            original_stack: Stack before expansion
            current_element_offset: Offset of the rule reference being expanded
            alternative_offset: Offset of one alternative of the referenced rule

        Returns:
            New stack tuple with the alternative incorporated
        """
        # Start with original stack minus the rule reference
        new_stack = list(original_stack[:-1])

        # Add continuation element if present (element after rule reference)
        next_element_offset = current_element_offset + 2
        if (
            self.grammar.grammar_encoding[next_element_offset]
            != END_OF_ALTERNATE_MARKER
        ):
            new_stack.append(next_element_offset)

        # Add first element of alternative if it's non-empty
        alternative_element_offset = alternative_offset + 1
        if (
            self.grammar.grammar_encoding[alternative_element_offset]
            != END_OF_ALTERNATE_MARKER
        ):
            new_stack.append(alternative_element_offset)

        return tuple(new_stack)

    def _expand_rule_reference(
        self, stack: Tuple[int], element_offset: int
    ) -> Set[Tuple[int]]:
        """
        Expand a rule reference into all its alternatives.

        Creates a new stack for each alternative of the referenced rule,
        then recursively expands those stacks in case they also start
        with rule references.

        Args:
            stack: Current parsing stack
            element_offset: Offset of the rule reference to expand

        Returns:
            Set of expanded stacks (one per alternative)
        """
        ref_rule_id = self._get_referenced_rule_id(element_offset)

        # Find start of the referenced rule's alternatives
        ref_subrule_offset = self.grammar.get_rule_offset(ref_rule_id) + 1

        new_stacks: Set[Tuple[int]] = set()

        # Process each alternative of the referenced rule
        while self.grammar.grammar_encoding[ref_subrule_offset] != END_OF_RULE_MARKER:
            alternative_stack = self._create_alternative_stack(
                stack, element_offset, ref_subrule_offset
            )

            # Recursively expand in case new stack also starts with rule reference
            new_stacks.update(self.expand_stack_head(alternative_stack))

            # Move to next alternative (skip over current alternative's length)
            ref_subrule_offset += self.grammar.grammar_encoding[ref_subrule_offset] + 1

        return new_stacks

    @lru_cache(maxsize=32768)
    def expand_stack_head(self, stack: Tuple[int]) -> Set[Tuple[int]]:
        """
        Expand stack by processing rule references at the top.

        Stacks represent the internal state of the pushdown automaton used for
        grammar recognition. When the top of the stack is a non-terminal (rule
        reference), we must expand it by replacing it with each alternative of
        the referenced rule, creating multiple possible parse paths.

        Args:
            stack: Current parsing stack (tuple of offsets into grammar encoding)

        Returns:
            Set of stacks after expansion. If the top element is not a rule
            reference, returns a set containing just the original stack.
        """
        # Empty stack cannot be expanded
        if len(stack) == 0:
            return {stack}

        # Get element at top of stack
        current_element_offset = stack[-1]

        # Terminal elements don't need expansion
        if not self._is_rule_reference(current_element_offset):
            return {stack}

        # Expand rule reference into all its alternatives
        return self._expand_rule_reference(stack, current_element_offset)
