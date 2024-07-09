from unittest import TestCase

from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer
from dataclasses import dataclass


@dataclass
class GeoQueryTestCase:
    name: str
    geo_query: str


valid_geo_query_sentences = [
    GeoQueryTestCase("simple_answer", "answer(smallest(state(all)))"),
    GeoQueryTestCase("state_id", "answer(highest(place(loc_2(stateid('hawaii')))))"),
    GeoQueryTestCase("river", "answer(river(all))"),
    GeoQueryTestCase("state", "answer(loc_1(major(river(all))))"),
    GeoQueryTestCase("next_to_2", "answer(state(next_to_2(stateid('texas'))))"),
    GeoQueryTestCase(
        "intersection",
        "answer(intersection(state(next_to_2(stateid('texas'))), loc_1(major(river(all)))))",
    ),
    GeoQueryTestCase("space in name", "answer(population_1(stateid('new york')))"),
    GeoQueryTestCase(
        "exclude",
        "answer(count(exclude(river(all), traverse_2(state(loc_1(capital(cityid('albany', _))))))))",
    ),
    GeoQueryTestCase(
        "city_id_with_state", "answer(population_1(cityid('washington', 'dc')))"
    ),
]

valid_geo_query_prefixes = [
    GeoQueryTestCase("empty_string", ""),
    GeoQueryTestCase("unbalanced_paranthesis", "answer(count(major(city(all"),
]

invalid_geo_query_sentences = [
    GeoQueryTestCase("no_answer", "highest(place(loc_2(stateid('kansas'))))"),
    GeoQueryTestCase("fake_country", "answer(major(city(loc_2(countryid('xx')))))"),
    GeoQueryTestCase("unexisting_function", "answer(population_2(stateid('hawaii')))"),
    GeoQueryTestCase("empty_operator", "answer(highest(place(loc_2())))"),
    GeoQueryTestCase("empty_paranthesis", "()"),
    GeoQueryTestCase(
        "missing_argument", "answer(intersection(state(next_to_2(stateid('texas'))), )"
    ),
]


class Test_parsing_geo_query_object(TestCase):
    def setUp(self):
        with open(f"examples/grammars/geo_query.ebnf", "r") as file:
            input_text = file.read()
        parsed_grammar = parse_ebnf(input_text)
        start_rule_id = parsed_grammar.symbol_table["root"]
        self.recognizer = StringRecognizer(
            parsed_grammar.grammar_encoding, start_rule_id
        )
        print("SetUp successfull!", flush=True)

    def test_valid_sentence(self):

        for geo_query_test_case in valid_geo_query_sentences:
            self.assertEqual(
                True,
                self.recognizer._accept_string(geo_query_test_case.geo_query),
                msg=f"Failed on {geo_query_test_case.name}, {geo_query_test_case.geo_query}",
            )
        for geo_query_test_case in (
            valid_geo_query_prefixes + invalid_geo_query_sentences
        ):
            self.assertEqual(
                False,
                self.recognizer._accept_string(geo_query_test_case.geo_query),
                msg=f"Failed on {geo_query_test_case.name}, {geo_query_test_case.geo_query}",
            )

    def test_valid_prefixes(self):
        for geo_query_test_case in valid_geo_query_sentences + valid_geo_query_prefixes:
            self.assertEqual(
                True,
                self.recognizer._accept_prefix(geo_query_test_case.geo_query),
                msg=f"Failed on {geo_query_test_case.name}, {geo_query_test_case.geo_query}",
            )

        for geo_query_test_case in invalid_geo_query_sentences:
            self.assertEqual(
                False,
                self.recognizer._accept_prefix(geo_query_test_case.geo_query),
                msg=f"Failed on {geo_query_test_case.name}, {geo_query_test_case.geo_query}",
            )
