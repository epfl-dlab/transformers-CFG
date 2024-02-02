from unittest import TestCase
from json import loads

import json
import logging

from tests.test_json import is_json_parsable

from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import GrammarRecognizer
# class Test(TestCase):
#     def test_minimal_json_array(self):
#         """
#         Test that we can load a JSON array
#         """
#         with open("examples/grammars/json_arr.ebnf", "r") as file:
#             input_text = file.read()
#         parsed_grammar = parse_ebnf(input_text)
#
#         start_rule_id = parsed_grammar.symbol_table["root"]
#         #
#         # res = recognizer._accept_string("12222", recognizer.stacks)
#
#         recognizer = GrammarRecognizer(parsed_grammar.grammar_encoding, start_rule_id)
#         jsons = [
#             "[]",
#             "[1]",
#             "[1,2]",
#             "[1,2,3]",
#         ]
#         for json in jsons:
#             recognizer = GrammarRecognizer(parsed_grammar.grammar_encoding, start_rule_id)
#             self.assertEqual(is_json_parsable(json), recognizer._accept_string(json, recognizer.stacks))