from unittest import TestCase

from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer
from dataclasses import dataclass


@dataclass
class OvernightTestCase:
    name: str
    overnight: str


valid_overnight_sentences = [
    OvernightTestCase(
        "simple_request", "(listValue (getProperty en.block.block1 width))"
    ),
    OvernightTestCase(
        "simple_filter",
        "(listValue (filter (getProperty (singleton en.block) !type) height = 3 en.inch))",
    ),
    OvernightTestCase(
        "count_values",
        "(listValue (countComparative (getProperty (singleton en.block) !type) shape >= 2))",
    ),
    OvernightTestCase(
        "ensure_property",
        "(listValue (filter (getProperty (singleton en.block) !type) (ensureNumericProperty width) <= (ensureNumericEntity 3 en.inch)))",
    ),
    OvernightTestCase(
        "above",
        "(listValue (filter (filter (getProperty (singleton en.block) !type) (reverse above) = en.block.block1) above = en.block.block1))",
    ),
    OvernightTestCase(
        "reverse right",
        "(listValue (filter (getProperty (singleton en.block) !type) right = (filter (getProperty (singleton en.block) !type) (reverse right) = en.block.block1)))",
    ),
    OvernightTestCase(
        "agg",
        "(listValue (superlative (getProperty (singleton en.block) !type) max (ensureNumericProperty length)))",
    ),
    OvernightTestCase(
        "nested_filters",
        "(listValue (filter (filter (getProperty (singleton en.block) !type) (reverse above) = en.block.block1) (reverse right) = en.block.block1))",
    ),
    OvernightTestCase(
        "shape",
        "(listValue (filter (getProperty (singleton en.block) !type) shape != en.shape.pyramid))",
    ),
    OvernightTestCase(
        "is_special",
        "(listValue (filter (filter (getProperty (singleton en.block) !type) is_special) left = en.block.block1))",
    ),
    OvernightTestCase(
        "two_blocks",
        "(listValue (filter (getProperty (singleton en.block) !type) left = (concat en.block.block1 en.block.block2)))",
    ),
    OvernightTestCase(
        "count_superlative",
        "(listValue (countSuperlative (getProperty (singleton en.block) !type) min (reverse above) (getProperty (singleton en.block) !type)))",
    ),
    OvernightTestCase(
        "long_query",
        "(listValue (filter (getProperty (singleton en.block) !type) (ensureNumericProperty height) > (ensureNumericEntity (getProperty en.block.block1 height))))",
    ),
    OvernightTestCase(
        "2_value",
        "(listValue (countComparative (getProperty (singleton en.block) !type) left < 2 (getProperty (singleton en.block) !type)))",
    ),
    OvernightTestCase(
        "concat_shapes",
        "(listValue (filter (getProperty (singleton en.block) !type) shape = (concat en.shape.pyramid en.shape.cube)))",
    ),
]


valid_overnight_prefixes = [
    OvernightTestCase("empty_string", ""),
    OvernightTestCase(
        "unbalanced_paranthesis", "(listValue (getProperty en.block.block1 width"
    ),
    OvernightTestCase("undefined_argument", "(listValue (getProperty en.block.block1"),
    OvernightTestCase(
        "left_comarisson",
        "(listValue (filter (getProperty (singleton en.block) !type) (ensureNumericProperty length) >=",
    ),
]

invalid_overnight_sentences = [
    OvernightTestCase(
        "unknown_property", "(listValue (getProperty en.block.block1 sparkliness))"
    ),
    OvernightTestCase("property", "(getProperty en.block.block1 width)"),
    OvernightTestCase("number_value", "3 en.inch"),
    OvernightTestCase(
        "extra_space", "(listValue ( getProperty en.block.block1 width))"
    ),
    OvernightTestCase("empty_operator", "(listValue (getProperty ))"),
    OvernightTestCase("empty_paranthesis", "()"),
    OvernightTestCase("missing_argument", "(listValue (getProperty en.block.block1 ))"),
    OvernightTestCase(
        "inexisting_shape",
        "(listValue (filter (getProperty (singleton en.block) !type) shape = (concat en.shape.pyramid en.shape.sphere)))",
    ),
]


class Test_parsing_overnight_object(TestCase):
    def setUp(self):
        with open(f"examples/grammars/overnight.ebnf", "r") as file:
            input_text = file.read()
        parsed_grammar = parse_ebnf(input_text)
        print("PARSED GRAMMAR:", parsed_grammar.grammar_encoding, flush=True)
        start_rule_id = parsed_grammar.symbol_table["root"]
        self.recognizer = StringRecognizer(
            parsed_grammar.grammar_encoding, start_rule_id
        )

    def test_valid_sentence(self):

        for overnight_test_case in valid_overnight_sentences:
            self.assertEqual(
                True,
                self.recognizer._accept_string(overnight_test_case.overnight),
                msg=f"Failed on {overnight_test_case.name}, {overnight_test_case.overnight}",
            )
        for overnight_test_case in (
            valid_overnight_prefixes + invalid_overnight_sentences
        ):
            self.assertEqual(
                False,
                self.recognizer._accept_string(overnight_test_case.overnight),
                msg=f"Failed on {overnight_test_case.name}, {overnight_test_case.overnight}",
            )

    def test_valid_prefixes(self):
        for overnight_test_case in valid_overnight_sentences + valid_overnight_prefixes:
            self.assertEqual(
                True,
                self.recognizer._accept_prefix(overnight_test_case.overnight),
                msg=f"Failed on {overnight_test_case.name}, {overnight_test_case.overnight}",
            )

        for overnight_test_case in invalid_overnight_sentences:
            self.assertEqual(
                False,
                self.recognizer._accept_prefix(overnight_test_case.overnight),
                msg=f"Failed on {overnight_test_case.name}, {overnight_test_case.overnight}",
            )
