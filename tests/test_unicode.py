import unittest
from unittest import TestCase

from transformers_cfg.recognizer import GrammarRecognizer

from transformers_cfg.parser import parse_ebnf
from tests.json_utils import is_json_parsable

import logging

logging.basicConfig(level=logging.DEBUG)


@unittest.skip("Skip for now")
class TestUnicode(TestCase):
    def test_minimal_json_object_with_unicode(self):
        """
        Test that we can load a JSON array
        """
        # json = '["foo", {"bar":["baz", null, 1.0, 2]}]'
        json = '{"foo": "bar", "三": "四"}'
        with open("examples/grammars/json.ebnf", "r") as file:
            input_text = file.read()
        parsed_grammar = parse_ebnf(input_text)

        start_rule_id = parsed_grammar.symbol_table["root"]

        recognizer = GrammarRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

        self.assertEqual(
            is_json_parsable(json),
            recognizer._accept_string(json, recognizer.stacks),
            f"Failed on {json}",
        )
