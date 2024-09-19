from unittest import TestCase

from transformers_cfg.parser import (
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
)
import logging

logger = logging.getLogger(__name__)


class Test(TestCase):
    def test_parse_name(self):
        rule = "root ::= [0-9]+"
        name, body = parse_name(rule)
        self.assertEqual("root", name, f"name: {name} != root")
        self.assertEqual(" ::= [0-9]+", body, f"body: {body} != ::= [0-9]+")

        rule_w_capital_letter = "ROOT ::= [0-9]+"
        name, body = parse_name(rule_w_capital_letter)
        self.assertEqual("ROOT", name, f"name: {name} != ROOT")
        self.assertEqual(" ::= [0-9]+", body, f"body: {body} != ::= [0-9]+")

        rule_with_name_starting_with_digit = "1root ::= [0-9]+"
        name, body = parse_name(rule_with_name_starting_with_digit)
        self.assertEqual("1root", name, f"name: {name} != 1root")
        self.assertEqual(" ::= [0-9]+", body, f"body: {body} != ::= [0-9]+")

        rule_with_underscore_in_name = "1_root ::= [0-9]+"
        name, body = parse_name(rule_with_underscore_in_name)
        self.assertEqual("1_root", name, f"name: {name} != 1_root")
        self.assertEqual(" ::= [0-9]+", body, f"body: {body} != ::= [0-9]+")

        rule_with_illegal_name = "root@1 ::= [0-9]+"
        name, body = parse_name(rule_with_illegal_name)
        self.assertEqual("root", name, f"name: {name} != root")
        self.assertEqual("@1 ::= [0-9]+", body, f"body: {body} != @1 ::= [0-9]+")

    def test_remove_leading_white_space(self):
        rule = " \t root ::= [0-9]+"
        _rule = remove_leading_white_space(rule, rm_leading_newline=False)
        self.assertEqual(
            rule.strip(), _rule, f"_rule: {_rule} != rule.strip(): {rule.strip()}"
        )
        self.assertEqual("root ::= [0-9]+", _rule, f"_rule: {_rule} != root ::= [0-9]+")

        rule_comment = "# comment"
        _rule = remove_leading_white_space(rule_comment, rm_leading_newline=False)
        self.assertEqual("", _rule, f"_rule: {_rule} != ''")

        # this function only removes leading white space and comments
        rule_end_with_comment = "root ::= [0-9]+ # comment"
        _rule = remove_leading_white_space(
            rule_end_with_comment, rm_leading_newline=False
        )

        self.assertEqual(
            rule_end_with_comment, _rule, f"_rule: {_rule} != {rule_end_with_comment}"
        )

        rule_comment_w_newline = "# comment\n root ::= [0-9]+"
        _rule = remove_leading_white_space(
            rule_comment_w_newline, rm_leading_newline=True
        )

        self.assertEqual("root ::= [0-9]+", _rule, f"_rule: {_rule} != root ::= [0-9]+")

        rulw_w_newline = "\n\n\n root ::= [0-9]+ \n rule2 ::= [0-9]+"
        _rule = remove_leading_white_space(rulw_w_newline, rm_leading_newline=False)
        self.assertTrue(
            _rule.startswith("\n"),
            f"_rule: {_rule} does not start with newline, but should",
        )

        _rule = remove_leading_white_space(rulw_w_newline, rm_leading_newline=True)
        self.assertFalse(
            _rule.startswith("\n"),
            f"_rule: {_rule} starts with newline, but should not",
        )

    def test__parse_rhs_negated_char_ranges(self):
        src = "[^a-z]"
        outbuf = []
        remaining_src = _parse_rhs_negated_char_ranges(src, outbuf)
        self.assertEqual(5, len(outbuf), f"len(outbuf): {len(outbuf)} != 5")

        self.assertListEqual([4, 0, 96, 122, 255], outbuf)
        self.assertEqual("", remaining_src, f"remaining_src: {remaining_src} != ''")

        src = "[^aeiou]"
        outbuf = []
        remaining_src = _parse_rhs_negated_char_ranges(src, outbuf)
        self.assertEqual(13, len(outbuf), f"len(outbuf): {len(outbuf)} != 13")
        self.assertListEqual(
            [12, 0, 96, 98, 100, 102, 104, 106, 110, 112, 116, 118, 255], outbuf
        )
        self.assertEqual("", remaining_src, f"remaining_src: {remaining_src} != ''")

        src = "[^0-9a-z]"
        outbuf = []

        remaining_src = _parse_rhs_negated_char_ranges(src, outbuf)
        self.assertEqual(7, len(outbuf), f"len(outbuf): {len(outbuf)} != 7")
        self.assertListEqual([6, 0, 47, 57, 96, 122, 255], outbuf)
        self.assertEqual("", remaining_src, f"remaining_src: {remaining_src} != ''")

    def test__parse_char_ranges(self):
        src = "[0-9]"
        outbuf = []

        start_idx = ord("0")
        end_idx = ord("9")

        remaining_src = _parse_rhs_char_ranges(src, outbuf)
        self.assertEqual(3, len(outbuf), f"len(outbuf): {len(outbuf)} != 3")
        self.assertListEqual([2, start_idx, end_idx], outbuf)

        self.assertEqual("", remaining_src, f"remaining_src: {remaining_src} != ''")

        src_enumerate = "[01234][0-9]"
        outbuf = []
        remaining_src = _parse_rhs_char_ranges(src_enumerate, outbuf)
        self.assertEqual(1 + 2 * 5, len(outbuf), f"len(outbuf): {len(outbuf)} != 11")
        self.assertListEqual([10, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52], outbuf)
        self.assertEqual(
            "[0-9]", remaining_src, f"remaining_src: {remaining_src} != ''"
        )

    def test__parse_rhs_any_char(self):
        src = "."
        outbuf = []

        remaining_src = _parse_rhs_any_char(src, outbuf)
        self.assertEqual(5, len(outbuf), f"len(outbuf): {len(outbuf)} != 1")
        self.assertListEqual([4, 0, 9, 11, 255], outbuf)
        self.assertEqual("", remaining_src, f"remaining_src: {remaining_src} != ''")

    def test__parse_literal_string(self):
        single_char_src = '"a"'
        outbuf = []

        remaining_src = _parse_rhs_literal_string(single_char_src, outbuf)
        self.assertEqual(3, len(outbuf), f"len(outbuf): {len(outbuf)} != 3")
        self.assertListEqual([2, ord("a"), ord("a")], outbuf)

        multi_char_src = '"abc"'
        outbuf = []

        remaining_src = _parse_rhs_literal_string(multi_char_src, outbuf)

        num_chars = len(multi_char_src) - 2
        self.assertEqual(
            num_chars * 3, len(outbuf), f"len(outbuf): {len(outbuf)} != {num_chars * 3}"
        )
        self.assertListEqual(
            [2, ord("a"), ord("a"), 2, ord("b"), ord("b"), 2, ord("c"), ord("c")],
            outbuf,
        )

        non_ascii_char_src = '"你"'
        outbuf = []

        remaining_src = _parse_rhs_literal_string(non_ascii_char_src, outbuf)
        self.assertEqual(3, len(outbuf), f"len(outbuf): {len(outbuf)} != 3")
        self.assertListEqual([2, ord("你"), ord("你")], outbuf)

    def test__parse_escape(self):
        escaped_char_src = '"\\n"'
        outbuf = []

        remaining_src = _parse_rhs_literal_string(escaped_char_src, outbuf)
        self.assertEqual(3, len(outbuf), f"len(outbuf): {len(outbuf)} != 3")
        self.assertListEqual([2, ord("\n"), ord("\n")], outbuf)

        escaped_backslash_src = '"\\\\"'
        outbuf = []

        remaining_src = _parse_rhs_literal_string(escaped_backslash_src, outbuf)
        self.assertEqual(3, len(outbuf), f"len(outbuf): {len(outbuf)} != 3")
        self.assertListEqual([2, ord("\\"), ord("\\")], outbuf)
        self.assertEqual("", remaining_src, f"remaining_src: {remaining_src} != ''")

        escaped_backslash_src = '"\\x5C"'
        outbuf = []

        remaining_src = _parse_rhs_literal_string(escaped_backslash_src, outbuf)
        self.assertEqual(3, len(outbuf), f"len(outbuf): {len(outbuf)} != 3")
        self.assertListEqual([2, ord("\\"), ord("\\")], outbuf)
        self.assertEqual("", remaining_src, f"remaining_src: {remaining_src} != ''")

    def test__parse_escape_unicode(self):
        # Test for 16-bit Unicode escape
        escaped_unicode_16_src = '"\\u20AC"'  # Unicode for Euro symbol
        outbuf = []
        remaining_src = _parse_rhs_literal_string(escaped_unicode_16_src, outbuf)
        self.assertEqual(3, len(outbuf), f"len(outbuf): {len(outbuf)} != 3")
        self.assertListEqual([2, ord("€"), ord("€")], outbuf)
        self.assertEqual("", remaining_src, f"remaining_src: {remaining_src} != ''")

        # Test for 32-bit Unicode escape
        escaped_unicode_32_src = '"\\U0001F600"'  # Unicode for grinning face emoji
        outbuf = []
        remaining_src = _parse_rhs_literal_string(escaped_unicode_32_src, outbuf)
        self.assertEqual(3, len(outbuf), f"len(outbuf): {len(outbuf)} != 3")
        self.assertListEqual([2, ord("😀"), ord("😀")], outbuf)
        self.assertEqual("", remaining_src, f"remaining_src: {remaining_src} != ''")

    def test_null(self):
        src = "root ::= "
        rhs_src = ""
        state = ParseState()
        state.symbol_table["root"] = 9
        _ = parse_rhs(
            state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
        )
        self.assertListEqual(
            [9, 1, END_OF_ALTERNATE_MARKER, END_OF_RULE_MARKER],
            state.grammar_encoding,
            f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}",
        )

    def test_parse_rhs(self):
        state = ParseState()
        outbuf = []
        src = 'root ::= "0"\n'
        rhs_src = '"0"\n'
        name, _ = parse_name(src)
        parse_simple_rhs(
            state=state, rhs=rhs_src, rule_name="root", outbuf=outbuf, is_nested=False
        )
        logging.debug(f"outbuf: {outbuf}")
        self.assertEqual(
            END_OF_ALTERNATE_MARKER,
            outbuf[-1],
            f"outbuf[-1]: {outbuf[-1]} != END_OF_RULE_MARKER",
        )

        state = ParseState()
        _ = parse_rhs(
            state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
        )
        self.assertEqual(
            9,
            state.grammar_encoding[0],
            f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}",
        )
        self.assertEqual(
            END_OF_RULE_MARKER,
            state.grammar_encoding[-1],
            f" The last symbol in the grammar encoding should be END_OF_RULE_MARKER, but got {state.grammar_encoding[-1]}",
        )
        self.assertEqual(
            END_OF_ALTERNATE_MARKER,
            state.grammar_encoding[-2],
            f" The second last symbol in the grammar encoding should be END_OF_SIMPLE_RULE_MARKER, but got {state.grammar_encoding[-2]}",
        )

        src = 'root ::= "2" | "3" | "4"'
        rhs_src = '"2" | "3" | "4"'

        state = ParseState()
        state.symbol_table["root"] = 9
        _ = parse_rhs(
            state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
        )
        self.assertEqual(
            9,
            state.grammar_encoding[0],
            f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}",
        )
        logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

        src = 'root ::= "2" | null | "4"'
        rhs_src = '"2" | '
        state = ParseState()
        state.symbol_table["root"] = 9
        _ = parse_rhs(
            state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
        )
        self.assertEqual(
            9,
            state.grammar_encoding[0],
            f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}",
        )
        logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

        src = 'root ::= "2" "3" "4"'
        rhs_src = '"2" "3" "4"'
        state = ParseState()
        state.symbol_table["root"] = 9
        _ = parse_rhs(
            state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
        )
        self.assertEqual(
            9,
            state.grammar_encoding[0],
            f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}",
        )
        logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

        src = 'root ::= "234"'
        rhs_src = '"234"'
        state = ParseState()
        state.symbol_table["root"] = 9
        _ = parse_rhs(
            state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
        )
        self.assertEqual(
            9,
            state.grammar_encoding[0],
            f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}",
        )
        logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

        src = "root ::= [234]"
        rhs_src = "[234]"
        state = ParseState()
        state.symbol_table["root"] = 9
        _ = parse_rhs(
            state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
        )
        self.assertEqual(
            9,
            state.grammar_encoding[0],
            f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}",
        )
        logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

        src = 'root ::= [234] | "5"'
        rhs_src = '[234] | "5"'
        state = ParseState()
        state.symbol_table["root"] = 9
        _ = parse_rhs(
            state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
        )
        self.assertEqual(
            9,
            state.grammar_encoding[0],
            f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}",
        )
        logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

        src = 'root ::= [234]"5"'
        rhs_src = '[234]"5"'
        state = ParseState()
        state.symbol_table["root"] = 9
        _ = parse_rhs(
            state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
        )
        self.assertEqual(
            9,
            state.grammar_encoding[0],
            f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}",
        )
        logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

        src = 'root ::= ("2" | "3" | "4") | ("5" | "6" | "7")'
        rhs_src = '("2" | "3" | "4") | ("5" | "6" | "7")'
        state = ParseState()
        state.symbol_table["root"] = 9
        _ = parse_rhs(
            state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False
        )
        logging.debug(f"state.grammar_encoding of {rhs_src}: {state.grammar_encoding}")

    def test__parse_symbol_reference(self):
        state = ParseState()
        outbuf = []
        _parse_rhs_symbol_reference("root", state, outbuf=outbuf)
        self.assertEqual(
            REF_RULE_MARKER, outbuf[0], f"outbuf[0]: {outbuf[0]} != REF_RULE_MARKER"
        )
        expected_symbol_id = state.symbol_table["root"]
        self.assertEqual(expected_symbol_id, outbuf[1], f"outbuf[1]: {outbuf[1]} != 0")
        # outbuf == [REF_RULE_MARKER, 0] == [1, 0]
        self.assertEqual(2, len(outbuf), f"len(outbuf): {len(outbuf)} != 2")

        # case where the symbol is already in the symbol table
        state = ParseState()
        state.symbol_table["root"] = 19
        outbuf = []
        _parse_rhs_symbol_reference("root", state, outbuf=outbuf)
        self.assertEqual(
            REF_RULE_MARKER, outbuf[0], f"outbuf[0]: {outbuf[0]} != REF_RULE_MARKER"
        )
        self.assertEqual(19, outbuf[1], f"outbuf[1]: {outbuf[1]} != 19")

    # def test__parse_rhs_grouping(self):
    #     src = "root ::= (\"0\" | \"1\")"
    #     outbuf = []
    #     name, _ = parse_name(src)
    #     state = ParseState()
    #     rhs_src = "(\"0\" | \"1\")"
    #     remaining_src = _parse_rhs_grouping(rhs_src, state=state, outbuf=outbuf,rule_name=name)
    #     self.assertEqual("", remaining_src, f"remaining_src: {remaining_src} != ''")
    #     logging.debug(f"outbuf: {outbuf}")
    #     logging.debug(f"grammar_encoding: {state.grammar_encoding}")
    #     import pdb; pdb.set_trace()

    # def test_parse_alternative_rhs(self):
    #     state = ParseState()
    #     outbuf = []
    #     src = 'root ::= "0" | "12"| "123"'
    #     rhs_src = '"0" | "12"| "123"'
    #     name, _ = parse_name(src)
    #     remaining_src = parse_simple_rhs(state=state, rhs=rhs_src, rule_name="root", outbuf=outbuf, is_nested=False)
    #     # parse_simple_rhs stops at the first |, so the remaining_src should start with |
    #     self.assertTrue(remaining_src.startswith("|"), f"remaining_src: {remaining_src} does not start with |")
    #
    #     remaining_src = parse_rhs(state=state, rhs=rhs_src, rule_name="root", rule_id=9, is_nested=False)
    #     self.assertEqual("", remaining_src, f"remaining_src: {remaining_src} != ''")
    #     logging.debug(f"grammar_encoding: {state.grammar_encoding}")
    #     self.assertEqual(9, state.grammar_encoding[0], f" The first symbol in the grammar encoding should be the rule id of root, which is 0, but got {state.grammar_encoding[0]}")
    #
    #     # split the grammar encoding into parts separated by END_OF_SIMPLE_RULE_MARKER
    #     parts = []
    #     part = []
    #     for sym in state.grammar_encoding[1:]: # skip the first symbol, which is the rule id of root
    #         if sym == END_OF_SIMPLE_RULE_MARKER:
    #             parts.append(part)
    #             part = []
    #         else:
    #             part.append(sym)
    #
    #     self.assertEqual(3, len(parts), f"The grammar encoding should have 3 parts, but got {len(parts)}")
    #     for part in parts:
    #         self.assertEqual(len(part), part[0], f"The first element of each part should be the length of the part, but got {part[0]}")

    def test__parse_rhs_repetition_operator_plus(self):
        src = 'root ::= "a"+'
        # translated_src = "root ::= ([0-9] root) "
        # translated_src = "root ::= ([0-9] | [1-2]) | "
        # logging.debug(f"\nsrc: {src}")
        outbuf = []
        name, _ = parse_name(src)
        state = ParseState()
        rhs_src = "[0-9]+"

        state = ParseState()
        outbuf2 = []
        parse_simple_rhs(
            state=state, rhs=rhs_src, rule_name="root", outbuf=outbuf2, is_nested=True
        )
        logging.debug(f"outbuf: {outbuf2}")
        logging.debug(f"parse_simple_rhs: {state.grammar_encoding}")

    # def test_plus_vs_star(self):
    #     src_plus = "root ::= [0-9]+"
    #     src_star = "root ::= [0-9]*"
    #     state_plus = parse_ebnf(src_plus)
    #     state_star = parse_ebnf(src_star)
    #
    #     self.assertEqual(state_plus.grammar_encoding, state_star.grammar_encoding, f"state_plus.grammar_encoding: {state_plus.grammar_encoding} != state_star.grammar_encoding: {state_star.grammar_encoding}")

    # def test__parse_rhs_grouping(self):
    #     src = "root ::= (\"0\" | \"1\")"
    #     outbuf = []
    #     name, _ = parse_name(src)
    #     state = ParseState()
    #     rhs_src = "(\"0\" | \"1\")"
    #     _parse_rhs_grouping(rhs_src, state=state, outbuf=outbuf,rule_name=name)
    #     logging.debug(f"outbuf: {outbuf}")
    #     import pdb; pdb.set_trace()
