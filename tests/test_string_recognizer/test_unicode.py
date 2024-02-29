import unittest
from unittest import TestCase

from transformers_cfg.recognizer import StringRecognizer, AcceptState

from transformers_cfg.parser import parse_ebnf
from tests.json_utils import is_json_parsable

import logging


class TestUnicode(TestCase):
    def test_accept_japanese(self):
        """
        Test that we can accept japanese characters
        """

        japanese = "こんにちは世界"
        with open("examples/grammars/japanese.ebnf", "r") as file:
            input_text = file.read()
        parsed_grammar = parse_ebnf(input_text)

        start_rule_id = parsed_grammar.symbol_table["root"]

        recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

        # accept_state = AcceptState.empty_state()

        self.assertTrue(recognizer._accept_prefix(japanese))

    def test_emoji(self):
        """
        Test that we can accept emoji
        """

        emoji = "😀😄😂"
        with open("examples/grammars/emoji.ebnf", "r") as file:
            input_text = file.read()
        parsed_grammar = parse_ebnf(input_text)

        start_rule_id = parsed_grammar.symbol_table["root"]

        recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

        self.assertTrue(recognizer._accept_prefix(emoji))
