import pytest
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.parser import parse_ebnf


@pytest.fixture(scope="module")
def japanese_recognizer():
    with open("examples/grammars/japanese.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)
    start_rule_id = parsed_grammar.symbol_table["root"]
    return StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)


@pytest.fixture(scope="module")
def emoji_recognizer():
    with open("examples/grammars/emoji.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)
    start_rule_id = parsed_grammar.symbol_table["root"]
    return StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)


def test_accept_japanese(japanese_recognizer):
    """
    Test that we can accept japanese characters
    """
    japanese = "こんにちは世界"
    assert japanese_recognizer._accept_prefix(japanese)


def test_emoji(emoji_recognizer):
    """
    Test that we can accept emoji
    """
    emoji = "😀😄😂"
    assert emoji_recognizer._accept_prefix(emoji)
