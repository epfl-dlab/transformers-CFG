import pytest
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer
from tests.json_utils import is_json_parsable

json_examples = {
    # Simple Nested Object
    "simple_nested": '{"name": "John", "age": 30, "address": {"street": "21 2nd Street", "city": "New York"}}',
    # Array of Objects
    "array_of_objects": '{"employees": [{"firstName": "John", "lastName": "Doe"}, {"firstName": "Anna", "lastName": "Smith"}]}',
    # Nested Arrays and Objects
    "nested_arrays_objects": '{"company": "OpenAI", "departments": [{"name": "Research", "members": [{"name": "Alice"}, {"name": "Bob"}]}, {"name": "Engineering", "members": [{"name": "Charlie"}]}]}',
    # Mixed Data Types
    "mixed_types": '{"name": "Alice", "age": 25, "isEmployee": true, "salary": null, "projects": ["NLP", "AI"]}',
    # Empty Object
    "empty_object": "{}",
    # Deeply Nested Object
    "deeply_nested": '{"level1": {"level2": {"level3": {"level4": {"message": "Deep"}}}}}',
    # Object with Numbers and Booleans
    "numbers_booleans": '{"temperature": 22.5, "isActive": false, "count": 10}',
    # Object with Array of Mixed Types
    "array_mixed_types": '{"data": [1, "two", true, null, {"nested": "object"}]}',
    # Complex Object with All Elements
    "complex_all_elements": '{"id": 101, "isActive": true, "info": {"name": "John Doe", "emails": ["john@example.com", "doe@example.com"], "address": {"city": "New York", "zip": "10001"}}, "tags": ["admin", "user"], "history": [{"login": "2023-01-01", "duration": 3600}, {"login": "2023-01-02", "duration": 2700}]}',
    # Object with Special Characters in Strings TODO fails
    # "escape_characters": '{"greeting": "Hello, \\"World\\"!", "path": "C:\\\\Program Files\\\\Test"}',
}


@pytest.fixture(scope="module")
def recognizer():
    with open("examples/grammars/json.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)
    start_rule_id = parsed_grammar.symbol_table["root"]
    recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)
    return recognizer


def test_minimal_json_object(recognizer):
    """
    Test that we can load a JSON object
    """
    json = '{"foo": "bar", "baz": "bat"}'

    assert is_json_parsable(json) == recognizer._accept_prefix(json)
    assert is_json_parsable(json) == recognizer._accept_string(json)

    prefix_json = json[: len(json) // 2]
    assert recognizer._accept_prefix(prefix_json)
    assert not recognizer._accept_string(prefix_json)


def test_systematic_examples(recognizer):
    for name, json_object in json_examples.items():
        assert is_json_parsable(json_object) == recognizer._accept_prefix(
            json_object
        ), f"Failed on {name}, {json_object}"
