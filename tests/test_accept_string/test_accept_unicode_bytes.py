import logging
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.parser import parse_ebnf


def test_accept_japanese():
    """
    Test that we can accept japanese characters
    """
    japanese = "こんにちは世界"
    with open("examples/grammars/japanese.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)

    start_rule_id = parsed_grammar.symbol_table["root"]

    recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

    bytes_japanese = bytes(japanese, "utf-8")
    logging.debug(f"bytes_japanese: {bytes_japanese} of length {len(bytes_japanese)}")
    # こんにちは世界

    head_bytes = bytes_japanese[:8]
    parsing_state = recognizer._update_state_with_bytes(head_bytes)

    # non empty stack means that the bytes were accepted
    assert len(parsing_state.stacks) > 0


def test_accept_japanese_progressive():
    #######################
    # Now consider the case of progressive matching
    #######################

    japanese = "こんにちは世界"
    with open("examples/grammars/japanese.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)

    start_rule_id = parsed_grammar.symbol_table["root"]

    recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

    bytes_japanese = bytes(japanese, "utf-8")
    logging.debug(f"bytes_japanese: {bytes_japanese} of length {len(bytes_japanese)}")

    byte_tokens = [bytes_japanese[i] for i in range(len(bytes_japanese))]
    # cast into bytes
    byte_tokens = [bytes([byte]) for byte in byte_tokens]

    parsing_state = recognizer.get_initial_parsing_state()

    for i, byte in enumerate(byte_tokens):
        parsing_state = recognizer._update_state_with_bytes(byte, parsing_state)
        assert len(parsing_state.stacks) > 0


def test_accept_emoji():
    """
    Test that we can accept emoji
    """
    emoji = "😀😄😂"
    with open("examples/grammars/emoji.ebnf", "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)

    start_rule_id = parsed_grammar.symbol_table["root"]

    recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

    bytes_emoji = bytes(emoji, "utf-8")
    logging.debug(f"bytes_emoji: {bytes_emoji} of length {len(bytes_emoji)}")
    # 😀😄😂

    parsing_state = recognizer._update_state_with_bytes(bytes_emoji)
    # non empty stack means that the bytes were accepted
    assert len(parsing_state.stacks) > 0
