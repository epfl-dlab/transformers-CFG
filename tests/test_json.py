from unittest import TestCase
from json import loads

import json

from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import GrammarRecognizer


def is_json_parsable(string):
    try:
        json.loads(string)
        return True
    except json.JSONDecodeError:
        return False
    except Exception as e:
        # You might want to handle or log other exceptions as well
        return False


# Example usage
json_string = '{"name": "John", "age": 30}'
print(is_json_parsable(json_string))  # This should return True

invalid_json_string = '{"name": "John", age: 30}'
print(is_json_parsable(invalid_json_string))  # This should return False

with open("examples/grammars/json.ebnf", "r") as file:
    input_text = file.read()
parsed_grammar = parse_ebnf(input_text)

start_rule_id = parsed_grammar.symbol_table["root"]
#
# res = recognizer._accept_string("12222", recognizer.stacks)


class Test(TestCase):
    # def test_minimal_json_array(self):
    #     """
    #     Test that we can load a JSON array
    #     """
    #     # json = '["foo", {"bar":["baz", null, 1.0, 2]}]'
    #     jsons = [
    #         "[]",
    #         "[1]",
    #         "[1,2]",
    #         "[1,2,3]",
    #     ]
    #     for json in jsons:
    #         recognizer = GrammarRecognizer(parsed_grammar.grammar_encoding, start_rule_id)
    #         self.assertEqual(is_json_parsable(json), recognizer._accept_string(json, recognizer.stacks))

    def test_minimal_json_object(self):
        """
        Test that we can load a JSON object
        """
        json = '{"foo": "bar", "baz": "bat"}'
        recognizer = GrammarRecognizer(parsed_grammar.grammar_encoding, start_rule_id)
        self.assertEqual(
            is_json_parsable(json), recognizer._accept_string(json, recognizer.stacks)
        )
