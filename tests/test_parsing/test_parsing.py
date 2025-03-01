import logging
from transformers_cfg.parser import (
    END_OF_GRAMMAR_MARKER,
    remove_leading_white_space,
    parse_name,
    _parse_rhs_negated_char_ranges,
    _parse_rhs_char_ranges,
    _parse_rhs_literal_string,
    _parse_rhs_any_char,
    ParseState,
    parse_simple_rhs,
    END_OF_RULE_MARKER,
    _parse_rhs_symbol_reference,
    REF_RULE_MARKER,
    parse_rhs,
    END_OF_ALTERNATE_MARKER,
    AlternativeElements,
    GrammarRule,
)

logger = logging.getLogger(__name__)


def test_parse_name():
    rule = "root ::= [0-9]+"
    name, body = parse_name(rule)
    assert name == "root", f"name: {name} != root"
    assert body == " ::= [0-9]+", f"body: {body} != ::= [0-9]+"

    rule_w_capital_letter = "ROOT ::= [0-9]+"
    name, body = parse_name(rule_w_capital_letter)
    assert name == "ROOT", f"name: {name} != ROOT"
    assert body == " ::= [0-9]+", f"body: {body} != ::= [0-9]+"

    rule_with_name_starting_with_digit = "1root ::= [0-9]+"
    name, body = parse_name(rule_with_name_starting_with_digit)
    assert name == "1root", f"name: {name} != 1root"
    assert body == " ::= [0-9]+", f"body: {body} != ::= [0-9]+"

    rule_with_underscore_in_name = "1_root ::= [0-9]+"
    name, body = parse_name(rule_with_underscore_in_name)
    assert name == "1_root", f"name: {name} != 1_root"
    assert body == " ::= [0-9]+", f"body: {body} != ::= [0-9]+"

    rule_with_illegal_name = "root@1 ::= [0-9]+"
    name, body = parse_name(rule_with_illegal_name)
    assert name == "root", f"name: {name} != root"
    assert body == "@1 ::= [0-9]+", f"body: {body} != @1 ::= [0-9]+"


def test_remove_leading_white_space():
    rule = " \t root ::= [0-9]+"
    _rule = remove_leading_white_space(rule, rm_leading_newline=False)
    assert _rule == rule.strip(), f"_rule: {_rule} != rule.strip(): {rule.strip()}"
    assert _rule == "root ::= [0-9]+", f"_rule: {_rule} != root ::= [0-9]+"

    rule_comment = "# comment"
    _rule = remove_leading_white_space(rule_comment, rm_leading_newline=False)
    assert _rule == "", f"_rule: {_rule} != ''"

    # this function only removes leading white space and comments
    rule_end_with_comment = "root ::= [0-9]+ # comment"
    _rule = remove_leading_white_space(rule_end_with_comment, rm_leading_newline=False)
    assert _rule == rule_end_with_comment, f"_rule: {_rule} != {rule_end_with_comment}"

    rule_comment_w_newline = "# comment\n root ::= [0-9]+"
    _rule = remove_leading_white_space(rule_comment_w_newline, rm_leading_newline=True)
    assert _rule == "root ::= [0-9]+", f"_rule: {_rule} != root ::= [0-9]+"

    rulw_w_newline = "\n\n\n root ::= [0-9]+ \n rule2 ::= [0-9]+"
    _rule = remove_leading_white_space(rulw_w_newline, rm_leading_newline=False)
    assert _rule.startswith(
        "\n"
    ), f"_rule: {_rule} does not start with newline, but should"

    _rule = remove_leading_white_space(rulw_w_newline, rm_leading_newline=True)
    assert not _rule.startswith(
        "\n"
    ), f"_rule: {_rule} starts with newline, but should not"


def test__parse_rhs_negated_char_ranges():
    src = "[^a-z]"
    alternative = AlternativeElements()
    remaining_src = _parse_rhs_negated_char_ranges(src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 5, f"len(outbuf): {len(outbuf)} != 5"
    assert outbuf == [4, 0, 96, 122, 255]
    assert remaining_src == "", f"remaining_src: {remaining_src} != ''"

    src = "[^aeiou]"
    alternative = AlternativeElements()
    remaining_src = _parse_rhs_negated_char_ranges(src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 13, f"len(outbuf): {len(outbuf)} != 13"
    assert outbuf == [12, 0, 96, 98, 100, 102, 104, 106, 110, 112, 116, 118, 255]
    assert remaining_src == "", f"remaining_src: {remaining_src} != ''"

    src = "[^0-9a-z]"
    alternative = AlternativeElements()
    remaining_src = _parse_rhs_negated_char_ranges(src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 7, f"len(outbuf): {len(outbuf)} != 7"
    assert outbuf == [6, 0, 47, 57, 96, 122, 255]
    assert remaining_src == "", f"remaining_src: {remaining_src} != ''"


def test__parse_char_ranges():
    src = "[0-9]"
    alternative = AlternativeElements()

    start_idx = ord("0")
    end_idx = ord("9")

    remaining_src = _parse_rhs_char_ranges(src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 3, f"len(outbuf): {len(outbuf)} != 3"
    assert outbuf == [2, start_idx, end_idx]
    assert remaining_src == "", f"remaining_src: {remaining_src} != ''"

    src_enumerate = "[01234][0-9]"
    alternative = AlternativeElements()
    remaining_src = _parse_rhs_char_ranges(src_enumerate, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 1 + 2 * 5, f"len(outbuf): {len(outbuf)} != 11"
    assert outbuf == [10, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52]
    assert remaining_src == "[0-9]", f"remaining_src: {remaining_src} != ''"


def test__parse_rhs_any_char():
    src = "."
    alternative = AlternativeElements()

    remaining_src = _parse_rhs_any_char(src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 5, f"len(outbuf): {len(outbuf)} != 1"
    assert outbuf == [4, 0, 9, 11, 255]
    assert remaining_src == "", f"remaining_src: {remaining_src} != ''"


def test__parse_literal_string():
    single_char_src = '"a"'
    alternative = AlternativeElements()

    remaining_src = _parse_rhs_literal_string(single_char_src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 3, f"len(outbuf): {len(outbuf)} != 3"
    assert outbuf == [2, ord("a"), ord("a")]

    multi_char_src = '"abc"'
    alternative = AlternativeElements()

    remaining_src = _parse_rhs_literal_string(multi_char_src, alternative)
    outbuf = alternative.serialize()[1:-1]

    num_chars = len(multi_char_src) - 2
    assert (
        len(outbuf) == num_chars * 3
    ), f"len(outbuf): {len(outbuf)} != {num_chars * 3}"
    assert outbuf == [
        2,
        ord("a"),
        ord("a"),
        2,
        ord("b"),
        ord("b"),
        2,
        ord("c"),
        ord("c"),
    ]

    non_ascii_char_src = '"你"'
    alternative = AlternativeElements()

    remaining_src = _parse_rhs_literal_string(non_ascii_char_src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 3, f"len(outbuf): {len(outbuf)} != 3"
    assert outbuf == [2, ord("你"), ord("你")]


def test__parse_escape():
    escaped_char_src = '"\\n"'
    alternative = AlternativeElements()

    remaining_src = _parse_rhs_literal_string(escaped_char_src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 3, f"len(outbuf): {len(outbuf)} != 3"
    assert outbuf == [2, ord("\n"), ord("\n")]

    escaped_backslash_src = '"\\\\"'
    alternative = AlternativeElements()

    remaining_src = _parse_rhs_literal_string(escaped_backslash_src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 3, f"len(outbuf): {len(outbuf)} != 3"
    assert outbuf == [2, ord("\\"), ord("\\")]
    assert remaining_src == "", f"remaining_src: {remaining_src} != ''"

    escaped_backslash_src = '"\\x5C"'
    alternative = AlternativeElements()

    remaining_src = _parse_rhs_literal_string(escaped_backslash_src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 3, f"len(outbuf): {len(outbuf)} != 3"
    assert outbuf == [2, ord("\\"), ord("\\")]
    assert remaining_src == "", f"remaining_src: {remaining_src} != ''"


def test__parse_escape_unicode():
    # Test for 16-bit Unicode escape
    escaped_unicode_16_src = '"\\u20AC"'  # Unicode for Euro symbol
    alternative = AlternativeElements()

    remaining_src = _parse_rhs_literal_string(escaped_unicode_16_src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 3, f"len(outbuf): {len(outbuf)} != 3"
    assert outbuf == [2, ord("€"), ord("€")]
    assert remaining_src == "", f"remaining_src: {remaining_src} != ''"

    # Test for 32-bit Unicode escape
    escaped_unicode_32_src = '"\\U0001F600"'  # Unicode for grinning face emoji
    alternative = AlternativeElements()

    remaining_src = _parse_rhs_literal_string(escaped_unicode_32_src, alternative)
    outbuf = alternative.serialize()[1:-1]
    assert len(outbuf) == 3, f"len(outbuf): {len(outbuf)} != 3"
    assert outbuf == [2, ord("😀"), ord("😀")]
    assert remaining_src == "", f"remaining_src: {remaining_src} != ''"


def test_null():
    src = "root ::= "
    rhs_src = ""
    state = ParseState()
    state.symbol_table["root"] = 9
    _ = parse_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
    )
    assert state.grammar_encoding == [
        9,
        1,
        END_OF_ALTERNATE_MARKER,
        END_OF_RULE_MARKER,
        END_OF_GRAMMAR_MARKER,
    ], f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}"


def test_parse_rhs():
    state = ParseState()
    src = 'root ::= "0"\n'
    rhs_src = '"0"\n'
    name, _ = parse_name(src)
    rule = GrammarRule(0, "root")
    parse_simple_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule=rule, is_nested=False
    )
    outbuf = rule.alternatives[0].serialize()
    logging.debug(f"outbuf: {outbuf}")
    assert (
        outbuf[-1] == END_OF_ALTERNATE_MARKER
    ), f"outbuf[-1]: {outbuf[-1]} != END_OF_RULE_MARKER"

    state = ParseState()
    _ = parse_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
    )
    assert (
        state.grammar_encoding[0] == 9
    ), f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}"
    assert (
        state.grammar_encoding[-2] == END_OF_RULE_MARKER
    ), f" The last symbol in the grammar encoding should be END_OF_RULE_MARKER, but got {state.grammar_encoding[-1]}"
    assert (
        state.grammar_encoding[-3] == END_OF_ALTERNATE_MARKER
    ), f" The second last symbol in the grammar encoding should be END_OF_SIMPLE_RULE_MARKER, but got {state.grammar_encoding[-2]}"

    src = 'root ::= "2" | "3" | "4"'
    rhs_src = '"2" | "3" | "4"'

    state = ParseState()
    state.symbol_table["root"] = 9
    _ = parse_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
    )
    assert (
        state.grammar_encoding[0] == 9
    ), f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}"
    logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

    src = 'root ::= "2" | null | "4"'
    rhs_src = '"2" | '
    state = ParseState()
    state.symbol_table["root"] = 9
    _ = parse_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
    )
    assert (
        state.grammar_encoding[0] == 9
    ), f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}"
    logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

    src = 'root ::= "2" "3" "4"'
    rhs_src = '"2" "3" "4"'
    state = ParseState()
    state.symbol_table["root"] = 9
    _ = parse_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
    )
    assert (
        state.grammar_encoding[0] == 9
    ), f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}"
    logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

    src = 'root ::= "234"'
    rhs_src = '"234"'
    state = ParseState()
    state.symbol_table["root"] = 9
    _ = parse_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
    )
    assert (
        state.grammar_encoding[0] == 9
    ), f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}"
    logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

    src = "root ::= [234]"
    rhs_src = "[234]"
    state = ParseState()
    state.symbol_table["root"] = 9
    _ = parse_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
    )
    assert (
        state.grammar_encoding[0] == 9
    ), f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}"
    logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

    src = 'root ::= [234] | "5"'
    rhs_src = '[234] | "5"'
    state = ParseState()
    state.symbol_table["root"] = 9
    _ = parse_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
    )
    assert (
        state.grammar_encoding[0] == 9
    ), f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}"
    logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

    src = 'root ::= [234]"5"'
    rhs_src = '[234]"5"'
    state = ParseState()
    state.symbol_table["root"] = 9
    _ = parse_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
    )
    assert (
        state.grammar_encoding[0] == 9
    ), f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}"
    logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

    src = 'root ::= ("2" | "3" | "4") | ("5" | "6" | "7")'
    rhs_src = '("2" | "3" | "4") | ("5" | "6" | "7")'
    state = ParseState()
    state.symbol_table["root"] = 9
    _ = parse_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
    )
    logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")


def test__parse_symbol_reference():
    state = ParseState()
    outbuf = []
    alternative = AlternativeElements()
    _parse_rhs_symbol_reference("root", state, alternative=alternative)
    outbuf = alternative.serialize()[1:-1]
    assert outbuf[0] == REF_RULE_MARKER, f"outbuf[0]: {outbuf[0]} != REF_RULE_MARKER"
    expected_symbol_id = state.symbol_table["root"]
    assert outbuf[1] == expected_symbol_id, f"outbuf[1]: {outbuf[1]} != 0"
    # outbuf == [REF_RULE_MARKER, 0] == [1, 0]
    assert len(outbuf) == 2, f"len(outbuf): {len(outbuf)} != 2"

    # case where the symbol is already in the symbol table
    state = ParseState()
    state.symbol_table["root"] = 19
    alternative = AlternativeElements()
    _parse_rhs_symbol_reference("root", state, alternative=alternative)
    outbuf = alternative.serialize()[1:-1]
    assert outbuf[0] == REF_RULE_MARKER, f"outbuf[0]: {outbuf[0]} != REF_RULE_MARKER"
    assert outbuf[1] == 19, f"outbuf[1]: {outbuf[1]} != 19"


def test__parse_rhs_repetition_operator_plus():
    src = 'root ::= "a"+'
    # translated_src = "root ::= ([0-9] root) "
    # translated_src = "root ::= ([0-9] | [1-2]) | "
    # logging.debug(f"\nsrc: {src}")
    outbuf = []
    name, _ = parse_name(src)
    state = ParseState()
    rhs_src = "[0-9]+"

    state = ParseState()
    rule = GrammarRule(0, "root")
    parse_simple_rhs(
        state=state, rhs=rhs_src, rule_name="root", rule=rule, is_nested=True
    )
    logging.debug(f"outbuf: {rule.serialize()}")
    logging.debug(f"parse_simple_rhs: {state.grammar_encoding}")
