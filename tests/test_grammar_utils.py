from unittest import TestCase
from transformers_cfg.grammar_utils import parse_name, remove_leading_white_space


class Test(TestCase):
    def test_parse_name(self):
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

    def test_remove_leading_white_space(self):
        rule = " \t root ::= [0-9]+"
        _rule = remove_leading_white_space(rule, rm_leading_newline=False)
        assert _rule == rule.strip(), f"_rule: {_rule} != rule.strip(): {rule.strip()}"
        assert _rule == "root ::= [0-9]+", f"_rule: {_rule} != root ::= [0-9]+"

        rule_comment = "# comment"
        _rule = remove_leading_white_space(rule_comment, rm_leading_newline=False)
        assert _rule == "", f"_rule: {_rule} != ''"

        # this function only removes leading white space and comments
        rule_end_with_comment = "root ::= [0-9]+ # comment"
        _rule = remove_leading_white_space(
            rule_end_with_comment, rm_leading_newline=False
        )

        assert (
            _rule == rule_end_with_comment
        ), f"_rule: {_rule} != {rule_end_with_comment}"

        rule_comment_w_newline = "# comment\n root ::= [0-9]+"
        _rule = remove_leading_white_space(
            rule_comment_w_newline, rm_leading_newline=True
        )
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
