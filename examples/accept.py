import argparse
import logging

from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer

logging.basicConfig(level=logging.DEBUG)


def main(args):

    with open(args.grammar_file_path, "r") as file:
        grammar_str = file.read()
    parsed_grammar = parse_ebnf(grammar_str)
    start_rule_id = parsed_grammar.symbol_table["root"]
    recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

    if args.mode == "prefix":
        result = recognizer._accept_prefix(args.sentence)
    else:
        result = recognizer._accept_string(args.sentence)

    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with grammar constraints."
    )
    parser.add_argument(
        "-g",
        "--grammar_file_path",
        type=str,
        required=True,
        help="Path to the grammar file (supports both relative and absolute paths)",
    )
    parser.add_argument(
        "-s", "--sentence", type=str, required=True, help="Prefix prompt for generation"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["prefix", "sentence"],
        default="prefix",
        help="Mode of operation, "
        "prefix mode accepts a prefix string, sentence mode only accepts a full sentence",
    )

    args = parser.parse_args()
    main(args)
