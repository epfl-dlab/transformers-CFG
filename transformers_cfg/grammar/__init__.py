"""
Grammar state management and recognition.

This module provides core components for managing grammar encoding,
parsing stacks, and character matching:

- GrammarState: Manages grammar encoding and rule lookups
- StackManager: Handles parsing stack operations and expansion
- CharacterMatcher: Tests if characters match grammar rules
- StringRecognizer: High-level orchestrator for string recognition (in parent module)

Usage:
    >>> from transformers_cfg.parser import parse_ebnf
    >>> from transformers_cfg.grammar import GrammarState, StackManager
    >>>
    >>> parsed = parse_ebnf('root ::= "hello" | "world"')
    >>> grammar_state = GrammarState(parsed.grammar_encoding, start_rule_id=0)
    >>> stack_manager = StackManager(grammar_state)
    >>> initial_stacks = stack_manager.create_initial_stacks()
"""

from .grammar_state import GrammarState
from .stack_manager import StackManager
from .character_matcher import CharacterMatcher

__all__ = ["GrammarState", "StackManager", "CharacterMatcher"]
