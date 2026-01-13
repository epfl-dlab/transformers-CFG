"""
Parsing stack management for grammar recognition.

This module provides the StackManager class which handles all parsing stack
operations for pushdown automaton-based grammar recognition. It manages stack
creation, expansion, and rule reference resolution with LRU caching for performance.

Key Components:
    StackManager: Manages parsing stacks representing possible parse states

Architecture:
    Grammar recognition uses a set-based pushdown automaton approach:
    - Each stack is a tuple of grammar element offsets
    - Multiple stacks represent alternative parse paths
    - Stack operations include:
      * Initialization: Create stacks from start rule alternatives
      * Expansion: Resolve rule references and follow alternatives
      * Advancement: Move to next element after consuming input

    Stack representation:
    stack = (element_offset_1, element_offset_2, ..., element_offset_n)
    where each offset points to a grammar element that expects the next input.

    Stack expansion flow:
    1. Check if stack head is a rule reference (REF_RULE_MARKER)
    2. If yes, replace with all alternatives from referenced rule
    3. If no (terminal), stack is already expanded
    4. Recursively expand any new rule references
    5. Cache result for this stack tuple

Example:
    >>> from transformers_cfg.grammar.stack_manager import StackManager
    >>> from transformers_cfg.grammar.grammar_state import GrammarState
    >>>
    >>> # Create manager with grammar state
    >>> manager = StackManager(grammar_state)
    >>>
    >>> # Initialize parsing stacks
    >>> initial_stacks = manager.create_initial_stacks()
    >>> print(f"Starting with {len(initial_stacks)} parse paths")
    >>>
    >>> # Expand a stack (e.g., after consuming a character)
    >>> if initial_stacks:
    ...     stack = list(initial_stacks)[0]
    ...     expanded = manager.expand_stack_head(stack)
    ...     print(f"Expansion yielded {len(expanded)} stacks")

Stack Operations:
    1. create_initial_stacks():
       - Called once to initialize parsing
       - Expands start rule into all possible starting stacks
       - Each alternative of start rule becomes a stack
       - Critical bug fixed: Now processes ALL alternatives, not just first

    2. expand_stack_head(stack):
       - Called after consuming input to advance parser
       - If stack head is rule reference, replace with rule alternatives
       - If stack head is terminal, return as-is
       - Cached via @lru_cache(maxsize=32768) for performance

Caching Strategy:
    Stack expansion uses LRU cache with 32768 entries because:
    - Same stacks recur frequently during generation
    - Expansion involves recursive grammar traversal (expensive)
    - Cache dramatically improves performance for complex grammars
    - Note: Cache persists in memory until explicit clearing

Integration:
    Used by StringRecognizer to manage parsing state as it consumes input
    characters/bytes. Works closely with GrammarState for rule lookups and
    CharacterMatcher for determining when to advance stacks.

Performance:
    - Initial stack creation: O(alternatives in start rule)
    - Stack expansion without cache: O(depth of rule references)
    - Stack expansion with cache: O(1) for cache hits
    - Memory: One cache entry per unique stack configuration encountered

See Also:
    - grammar_state.py: Provides grammar encoding and rule offset lookups
    - character_matcher.py: Determines when to advance stacks
    - ../recognizer.py: Uses StackManager for parsing orchestration
"""

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
