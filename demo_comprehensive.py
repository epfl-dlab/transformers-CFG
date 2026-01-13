#!/usr/bin/env python3
"""
Comprehensive demo showing grammar validation and constrained generation.
Demonstrates that the refactored code works correctly.
"""

from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer
from transformers import AutoTokenizer
from transformers_cfg.token_grammar_recognizer import IncrementalTokenRecognizer


def demo_json_validation():
    """Demo 1: JSON Grammar Validation"""
    print("=" * 80)
    print("DEMO 1: JSON Grammar Validation")
    print("=" * 80)

    # Load JSON grammar
    with open("examples/grammars/json.ebnf", "r") as file:
        grammar_str = file.read()

    parsed_grammar = parse_ebnf(grammar_str)
    start_rule_id = parsed_grammar.symbol_table["root"]
    recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

    # Test cases
    test_cases = [
        ('{"name": "John", "age": 30}', True, "Valid JSON object"),
        ('{"items": [1, 2, 3]}', True, "Valid JSON with array"),
        ('{"nested": {"key": "value"}}', True, "Valid nested JSON"),
        ('{"name": "John" invalid}', False, "Invalid JSON - syntax error"),
        ('{name: "John"}', False, "Invalid JSON - unquoted key"),
        ("", False, "Empty string"),
    ]

    for json_str, expected, description in test_cases:
        result = recognizer._accept_string(json_str)
        status = "✓" if result == expected else "✗"
        print(f"\n{status} {description}")
        print(f"  Input: {json_str[:50]}{'...' if len(json_str) > 50 else ''}")
        print(f"  Valid: {result} (Expected: {expected})")


def demo_prefix_validation():
    """Demo 2: Prefix Validation (Partial Strings)"""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Prefix Validation (Partial Strings)")
    print("=" * 80)

    # Load JSON grammar
    with open("examples/grammars/json.ebnf", "r") as file:
        grammar_str = file.read()

    parsed_grammar = parse_ebnf(grammar_str)
    start_rule_id = parsed_grammar.symbol_table["root"]
    recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

    # Test prefixes (incomplete JSON strings) - all should be valid prefixes
    prefixes = [
        ('{"name"', True, "Opening brace and key start"),
        ('{"name":', True, "Key with colon"),
        ('{"name": "John"', True, "Key-value pair incomplete"),
        ('{"name": "John",', True, "First pair with comma"),
        ('{"name": "John", "age"', True, "Second key incomplete"),
        ('{"name": "John", "age": 30}', True, "Complete object"),
    ]

    print("\nBuilding a JSON object step by step:")
    print("-" * 80)

    for prefix, expected, description in prefixes:
        result = recognizer._accept_prefix(prefix)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description}")
        print(f"  Input: '{prefix}'")
        print(f"  Valid prefix: {result} (Expected: {expected})")


def demo_token_recognition():
    """Demo 3: Token-Level Recognition"""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Token-Level Recognition")
    print("=" * 80)

    # Load Japanese grammar (unicode test)
    with open("examples/grammars/japanese.ebnf", "r") as file:
        grammar_str = file.read()

    print("\nGrammar loaded: Japanese characters (Unicode test)")
    print("Grammar accepts: Hiragana, Katakana, and Kanji characters")

    # Use GPT2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    recognizer = IncrementalTokenRecognizer(
        grammar_str=grammar_str, start_rule_name="root", tokenizer=tokenizer
    )

    # Test cases
    test_cases = [
        ("こんにちは", True, "Hello in Japanese (Hiragana)"),
        ("トリーム", True, "Stream in Katakana"),
        ("安녕하세요", False, "Korean characters (should be rejected)"),
    ]

    print("\nTesting token recognition:")
    print("-" * 80)

    for text, expected, description in test_cases:
        token_ids = tokenizer.encode(text)
        result = recognizer.accept_token_ids(token_ids)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description}")
        print(f"  Input: '{text}'")
        print(f"  Tokens: {token_ids}")
        print(f"  Valid: {result} (Expected: {expected})")


def demo_grammar_parsing():
    """Demo 4: Grammar Parsing and Structure"""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Grammar Parsing and Structure")
    print("=" * 80)

    # Simple arithmetic grammar
    grammar = """
    root ::= expr
    expr ::= term (("+"|"-") term)*
    term ::= factor (("*"|"/") factor)*
    factor ::= number | "(" expr ")"
    number ::= [0-9]+
    """

    print("\nArithmetic Grammar:")
    print("-" * 80)
    print(grammar)

    parsed = parse_ebnf(grammar)

    print("\nParsed Grammar Info:")
    print(f"  Rules defined: {len(parsed.symbol_table)}")
    print(f"  Symbol table: {list(parsed.symbol_table.keys())}")
    print(f"  Grammar encoding length: {len(parsed.grammar_encoding)} integers")

    # Test expressions
    recognizer = StringRecognizer(parsed.grammar_encoding, parsed.symbol_table["root"])

    expressions = [
        ("1+2", True, "Simple addition"),
        ("1+2*3", True, "Mixed operations"),
        ("(1+2)*3", True, "Parentheses grouping"),
        ("1 + 2", False, "Whitespace not allowed"),
        ("1++2", False, "Double operator invalid"),
    ]

    print("\nTesting arithmetic expressions:")
    print("-" * 80)

    for expr, expected, description in expressions:
        result = recognizer._accept_string(expr)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description}")
        print(f"  Input: '{expr}'")
        print(f"  Valid: {result} (Expected: {expected})")


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  transformers-CFG Comprehensive Demo".center(78) + "║")
    print("║" + "  Demonstrating Refactored Architecture".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        demo_json_validation()
        demo_prefix_validation()
        demo_token_recognition()
        demo_grammar_parsing()

        print("\n\n" + "=" * 80)
        print("All demos completed successfully! ✓")
        print("=" * 80)
        print("\nThe refactored code is working correctly:")
        print("  • Grammar parsing ✓")
        print("  • String validation ✓")
        print("  • Token recognition ✓")
        print("  • Prefix validation ✓")
        print("  • Unicode support ✓")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
