import argparse
import logging
import sys
from abc import ABC
from typing import List, Tuple
from functools import cached_property
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

END_OF_ALTERNATE_MARKER = 0
END_OF_RULE_MARKER = 0
END_OF_GRAMMAR_MARKER = 0xFFFF
TO_BE_FILLED_MARKER = 0
REF_RULE_MARKER = 1


########################
# EBNF Grammar Parsing #
########################


class Codable(ABC):
    def serialize(self) -> List[int]:
        pass

    @classmethod
    def deserialize(cls, data: List[int]) -> "Codable":
        pass


@dataclass
class GrammarElement(Codable):
    def is_terminated(self) -> bool:
        raise NotImplementedError()
    

@dataclass
class TerminatedElement(GrammarElement):
    ranges: List[Tuple[int, int]]

    def is_terminated(self) -> bool:
        return True
    
    def serialize(self) -> List[int]:
        outbuf = [len(self.ranges) * 2]
        for range in self.ranges:
            outbuf.extend(range)
        return outbuf
    
    @classmethod
    def from_range(cls, start: int, end: int) -> "GrammarElement":
        return cls([(start, end)])


@dataclass
class ReferenceElement(GrammarElement):
    reference_id: int

    def is_terminated(self) -> bool:
        return False
    
    def serialize(self) -> List[int]:
        return [REF_RULE_MARKER, self.reference_id]
    

@dataclass
class AlternativeElements(Codable):
    elements: List[GrammarElement] = field(default_factory=list)

    def add_element(self, element: GrammarElement) -> None:
        self.elements.append(element)

    def serialize(self) -> List[int]:
        outbuf = [TO_BE_FILLED_MARKER]
        for element in self.elements:
            outbuf.extend(element.serialize())
        outbuf[0] = len(outbuf)
        outbuf.append(END_OF_ALTERNATE_MARKER)
        return outbuf


@dataclass
class GrammarRule(Codable):
    id: int
    name: str
    alternatives: List[AlternativeElements] = field(default_factory=list)

    def add_alternative(self, alternative: AlternativeElements) -> None:
        self.alternatives.append(alternative)

    def add_empty_alternative(self) -> AlternativeElements:
        new_alternative = AlternativeElements()
        self.alternatives.append(new_alternative)
        return new_alternative

    def serialize(self) -> List[int]:
        outbuf = [self.id]
        for alternative in self.alternatives:
            outbuf.extend(alternative.serialize())
        outbuf.append(END_OF_RULE_MARKER)
        return outbuf


class ParseState:
    def __init__(self):
        self.symbol_table = {}
        self.grammar_rules: List[GrammarRule] = []

    def add_rule(self, rule: GrammarRule) -> None:
        self.grammar_rules.append(rule)

    def get_rule_by_id(self, id: int) -> GrammarRule:
        for rule in self.grammar_rules:
            if rule.id == id:
                return rule
        raise ValueError(f"No rule with id {id} found")

    @cached_property
    def grammar_encoding(self) -> List[int]: # old name: out_grammar
        outbuf = []
        for rule in self.grammar_rules:
            outbuf.extend(rule.serialize())
        outbuf.append(END_OF_GRAMMAR_MARKER)
        return outbuf

    def print(self, file=sys.stdout):
        print_grammar(file, self)


def get_symbol_id(state: ParseState, symbol_name: str) -> int:
    if symbol_name not in state.symbol_table:
        state.symbol_table[symbol_name] = len(state.symbol_table)
    return state.symbol_table[symbol_name]


def generate_symbol_id(state: ParseState, base_name: str) -> int:
    next_id = len(state.symbol_table)
    state.symbol_table[base_name + "_" + str(next_id)] = next_id
    return next_id


def is_word_char(c: str) -> bool:
    """
    Check if a char is  a-z, A-Z, 0-9, -, _, i.e., chars allowed as rule names
    Returns:

    """
    return c.isalnum() or c == "-" or c == "_"


def hex_to_int(c: str) -> int:
    """
    Convert a hex char to int, c should be in the range of 0-9, a-f, A-F
    case insensitive
    Args:
        c:  a hex char
    Returns:
        int: the int value of the hex char
    """
    if c.isdigit():
        return int(c)
    elif "a" <= c.lower() <= "f":
        return ord(c.lower()) - ord("a") + 10
    return -1


def remove_leading_white_space(src, rm_leading_newline):
    """
    Skips over whitespace and comments in the input string.

    This function processes the input string, skipping over any spaces, tabs,
    and content following a '#' character, which denotes a comment. The parsing
    of a comment continues until the end of the line (denoted by newline characters
    '\r' or '\n'). If the 'rm_leading_newline' parameter is set to False, the function
    will stop processing and return the remaining string upon encountering a
    newline character, otherwise it will skip over newline characters as well.

    Parameters:
    src (str): The input string to be processed.
    rm_leading_newline (bool): A flag indicating whether encountering a newline character
                       should stop the parsing (False) or if it should be skipped (True).

    Returns:
    str: The remaining portion of the input string after skipping whitespace and comments.
    """
    pos = 0
    while pos < len(src) and (src[pos].isspace() or src[pos] == "#"):
        if src[pos] == "#":
            while pos < len(src) and src[pos] not in ("\r", "\n"):
                pos += 1
        else:
            if not rm_leading_newline and src[pos] in ("\r", "\n"):
                break
            pos += 1
    return src[pos:]


def parse_name(src: str) -> Tuple[str, str]:
    """
    parse the leading name from the input string
    Args:
        src:  the input grammar string

    Returns:
        name, remaining_src
    """
    pos = 0
    while pos < len(src) and is_word_char(src[pos]):
        pos += 1
    if pos == 0:
        raise RuntimeError("expecting name at " + src)
    return src[:pos], src[pos:]


def parse_char(src: str) -> Tuple[str, str]:
    """
    parse the leading char from the input string
    :param src:
    :return: char, remaining_src
    """

    # if we have a backslash, it's maybe an escape
    if src[0] == "\\":
        esc = src[1]
        if esc == "x":
            first = hex_to_int(src[2])
            if first > -1:
                second = hex_to_int(src[3])
                if second > -1:
                    return chr((first << 4) + second), src[4:]
            raise RuntimeError("expecting \\xNN at " + src)
        elif esc == "u":
            if len(src) >= 6:
                hex_value = src[2:6]
                if all(c in "0123456789ABCDEFabcdef" for c in hex_value):
                    return chr(int(hex_value, 16)), src[6:]
                raise RuntimeError("expecting \\uXXXX at " + src)
            raise RuntimeError("incomplete \\uXXXX escape at " + src)
        elif esc == "U":
            if len(src) >= 10:
                hex_value = src[2:10]
                if all(c in "0123456789ABCDEFabcdef" for c in hex_value):
                    return chr(int(hex_value, 16)), src[10:]
                raise RuntimeError("expecting \\UXXXXXXXX at " + src)
            raise RuntimeError("incomplete \\UXXXXXXXX escape at " + src)
        elif esc in ('"', "[", "]"):
            return esc, src[2:]
        elif esc == "r":
            return "\r", src[2:]
        elif esc == "n":
            return "\n", src[2:]
        elif esc == "t":
            return "\t", src[2:]
        elif esc == "\\":
            return "\\", src[2:]
        raise RuntimeError("unknown escape at " + src)
    elif src:
        return src[0], src[1:]
    raise RuntimeError("unexpected end of input")


def _parse_rhs_literal_string(src: str, alternative: AlternativeElements) -> str:
    assert src[0] == '"', f"rule should start with '\"', but got {src[0]}"
    remaining_src = src[1:]

    # advance until we get an end quote or run out of input
    while remaining_src and remaining_src[0] != '"':
        char, remaining_src = parse_char(remaining_src)
        alternative.add_element(TerminatedElement([(ord(char), ord(char))]))

    # in case we ran out of input before finding the end quote
    if not remaining_src:
        raise RuntimeError(f"expecting an end quote at {src},but not found")

    # remove the end quote and return the remaining string
    return remaining_src[1:]


def _parse_rhs_negated_char_ranges(src: str, alternative: AlternativeElements) -> str:
    assert src[:2] == "[^", f"rule should start with '[^', but got {src[:2]}"
    remaining_src = src[2:]
    neg_outbuf = []
    while remaining_src and remaining_src[0] != "]":
        char, remaining_src = parse_char(remaining_src)

        neg_outbuf.append(ord(char))
        if remaining_src[0] == "-" and remaining_src[1] != "]":
            endchar_pair, remaining_src = parse_char(remaining_src[1:])

            neg_outbuf.extend(range(ord(char) + 1,  ord(endchar_pair)))
        else:
            # This is the case for enumerate, e.g., [^0123456789], [^abcdef]
            # Each char is considered as a range of itself, i.e., c-c
            neg_outbuf.append(ord(char))
    if not remaining_src:
        raise RuntimeError(
            f"expecting an ] at {src},but not found, is the char range closed?"
        )
    
    # Compute allowed chars ranges
    neg_outbuf = [-1] + sorted(set(neg_outbuf)) + [0xFF + 1] # min ord, ..., max ord
    
    # Generate allowed ranges
    ranges = []
    for start, end in zip(neg_outbuf[:-1], neg_outbuf[1:]):
        allowed_start = start + 1
        allowed_end = end - 1
        if allowed_start <= allowed_end:
            ranges.append((allowed_start, allowed_end))
    
    alternative.add_element(TerminatedElement(ranges))
    return remaining_src[1:]


def _parse_rhs_char_ranges(src: str, alternative: AlternativeElements) -> str:
    assert src[0] == "[", f"rule should start with '[', but got {src[0]}"
    remaining_src = src[1:]
    ranges = []
    while remaining_src and remaining_src[0] != "]":
        char, remaining_src = parse_char(remaining_src)

        if remaining_src[0] == "-" and remaining_src[1] != "]":
            endchar_pair, remaining_src = parse_char(remaining_src[1:])
            ranges.append((ord(char), ord(endchar_pair)))
        else:
            # This is the case for enumerate, e.g., [0123456789], [abcdef]
            # Each char is considered as a range of itself, i.e., c-c
            ranges.append((ord(char), ord(char)))
    if not remaining_src:
        raise RuntimeError(
            f"expecting an ] at {src},but not found, is the char range closed?"
        )
    alternative.add_element(TerminatedElement(ranges))
    return remaining_src[1:]


def _parse_rhs_any_char(src: str, alternative: AlternativeElements) -> str:
    assert src[0] == ".", f"rule should start with '.', but got {src[0]}"
    remaining_src = src[1:]
    # The only symbol not allowed is '\n'
    alternative.add_element(TerminatedElement([(0, ord('\n') - 1), (ord('\n') + 1, 0xFF)]))
    return remaining_src


def _parse_rhs_symbol_reference(src: str, state: ParseState, alternative: AlternativeElements) -> str:
    assert is_word_char(src[0]), f"rule should start with a word char, but got {src[0]}"
    name, remaining_src = parse_name(src)
    ref_rule_id = get_symbol_id(state, name)
    alternative.add_element(ReferenceElement(ref_rule_id))
    return remaining_src


def _parse_rhs_grouping(
    remaining_src: str, state: ParseState, rule_name: str, alternative: AlternativeElements
) -> str:
    assert (
        remaining_src[0] == "("
    ), f"rule should start with '(', but got {remaining_src[0]}"
    remaining_src = remove_leading_white_space(remaining_src[1:], True)
    # parse nested alternates into synthesized rule
    synthetic_rule_id = generate_symbol_id(state, rule_name)
    remaining_src = parse_rhs(state, remaining_src, rule_name, synthetic_rule_id, True)
    # output reference to synthesized rule
    alternative.add_element(ReferenceElement(synthetic_rule_id))

    if not remaining_src or remaining_src[0] != ")":
        raise RuntimeError("expecting ')' at " + remaining_src)
    return remaining_src[1:]


def _parse_rhs_repetition_operators(
    remaining_src: str,
    state: ParseState,
    rule_name: str,
    alternative: AlternativeElements,
) -> str:
    assert remaining_src[0] in (
        "*",
        "+",
        "?",
    ), f"rule should start with '*', '+', or '?', but got {remaining_src[0]}"

    # apply transformation to previous symbol (last_sym_start -
    # end) according to rewrite rules:
    # S* --> S' ::= S S' |
    # S+ --> S' ::= S S' | S
    # S? --> S' ::= S |
    sub_rule_id = generate_symbol_id(state, rule_name)
    sub_rule = GrammarRule(sub_rule_id, f"{rule_name}_{sub_rule_id}")
    sub_rule_first_alternative = sub_rule.add_empty_alternative()
    # add preceding symbol to generated rule
    sub_rule_first_alternative.add_element(alternative.elements[-1])
    if remaining_src[0] in ("*", "+"):
        # cause generated rule to recurse
        sub_rule_first_alternative.add_element(ReferenceElement(sub_rule_id))
    sub_rule_second_alternative = AlternativeElements()
    sub_rule.add_alternative(sub_rule_second_alternative)
    if remaining_src[0] == "+":
        # add preceding symbol as alternate only for '+'
        sub_rule_second_alternative.add_element(alternative.elements[-1])

    state.grammar_rules.append(sub_rule)
    alternative.elements[-1] = ReferenceElement(sub_rule_id)
    return remaining_src[1:]


def _parse_rhs_numbered_repetition_operators(
    remaining_src: str,
    state: ParseState,
    rule_name: str,
    alternative: AlternativeElements,
) -> str:
    assert remaining_src[0] == "{", f"rule should start with '{{', but got {remaining_src[0]}"

    # parse numbers
    closing_brace_idx = remaining_src.find("}")
    numbers_src = remaining_src[1:closing_brace_idx]
    n_src, m_src = numbers_src.split(",") if "," in numbers_src else (numbers_src, numbers_src) # {n} -> {n, n}
    n = int(n_src) if n_src else 0
    m = int(m_src) if m_src else None

    # rules:
    # S{n} = S{n, n} --> S' ::= S S S ... S (n times)
    # S{n, m} --> S' ::= S S S ... S (n times) | S S S ... S (n + 1 times) | ... | S S S ... S (m times)
    # S{n,} --> S' ::= S S S ... S+ (n times)
    # S{,m} = S{0, m} --> S' ::= S | S S | S S S | ... | S S S ... S (m times)
    if not m: n -= 1 # remove the last S to replace with S+

    sub_rule_id = generate_symbol_id(state, rule_name)
    sub_rule = GrammarRule(sub_rule_id, f"{rule_name}_{sub_rule_id}")

    for i in range(max(n, 1), m + 1 if m else n + 1):
        sub_rule_alternative = AlternativeElements()
        sub_rule.add_alternative(sub_rule_alternative)
        for _ in range(i):
            sub_rule_alternative.add_element(alternative.elements[-1])
        if not m:
            sub_rule_alternative.add_element(ReferenceElement(sub_rule_id))

    if not m:
        sub_rule_alternative = AlternativeElements()
        sub_rule.add_alternative(sub_rule_alternative)
        sub_rule_alternative.add_element(alternative.elements[-1])

    state.grammar_rules.append(sub_rule)
    alternative.elements[-1] = ReferenceElement(sub_rule_id)
    return remaining_src[closing_brace_idx + 1:]


def parse_simple_rhs(state: ParseState, rhs: str, rule_name: str, rule: GrammarRule, is_nested: bool) -> str:
    remaining_rhs = rhs
    alternative = AlternativeElements()

    while remaining_rhs:
        if remaining_rhs[0] == '"':
            # literal string
            remaining_rhs = _parse_rhs_literal_string(remaining_rhs, alternative)
        elif remaining_rhs[:2] == "[^":
            # negated char range(s)
            remaining_rhs = _parse_rhs_negated_char_ranges(remaining_rhs, alternative)
        elif remaining_rhs[0] == "[":
            # char range(s)
            remaining_rhs = _parse_rhs_char_ranges(remaining_rhs, alternative)
        elif remaining_rhs[0] == ".":
            # any char
            remaining_rhs = _parse_rhs_any_char(remaining_rhs, alternative)
        elif is_word_char(remaining_rhs[0]):
            # rule reference
            remaining_rhs = _parse_rhs_symbol_reference(remaining_rhs, state, alternative)
        elif remaining_rhs[0] == "(":
            # grouping
            remaining_rhs = _parse_rhs_grouping(remaining_rhs, state, rule_name, alternative)
        elif remaining_rhs[0] in ("*", "+", "?", "{"):
            # repetition operator
            if remaining_rhs[0] == "{":
                remaining_rhs = _parse_rhs_numbered_repetition_operators(
                    remaining_rhs, state, rule_name, alternative
                )
            else:
                remaining_rhs = _parse_rhs_repetition_operators(
                    remaining_rhs, state, rule_name, alternative
                )
        else:
            # case for newline, i.e., end of rule
            assert remaining_rhs[0] in [
                "\n",
                "|",
                ")",
            ], f"rule should end with newline or '|', but got {remaining_rhs[0]}"
            # we break here so that we call parse_rule again to parse the next rule
            break
        # Here we do not rm newline deliberately so that we know the rhs is ended
        remaining_rhs = remove_leading_white_space(
            remaining_rhs, rm_leading_newline=is_nested
        )

    rule.add_alternative(alternative)
    return remaining_rhs


def parse_rhs(state: ParseState, rhs: str, rule_name: str, rule_id: int, is_nested: bool) -> str:
    rule = GrammarRule(rule_id, rule_name)
    remaining_rhs = parse_simple_rhs(state, rhs, rule_name, rule, is_nested)
    while remaining_rhs and remaining_rhs[0] == "|":
        remaining_rhs = remove_leading_white_space(remaining_rhs[1:], True)
        remaining_rhs = parse_simple_rhs(
            state, remaining_rhs, rule_name, rule, is_nested
        )

    state.grammar_rules.append(rule)
    return remaining_rhs


def parse_rule(state: ParseState, rule_text: str) -> str:
    name, remaining_rule_text = parse_name(rule_text)
    remaining_rule_text = remove_leading_white_space(remaining_rule_text, False)
    # check if the rule is already defined, TODO: what will happen if the rule is already defined?
    rule_id = get_symbol_id(state, name)

    if remaining_rule_text[:3] != "::=":
        raise RuntimeError("expecting ::= at " + remaining_rule_text)
    remaining_rule_text = remove_leading_white_space(remaining_rule_text[3:], True)

    remaining_rule_text = parse_rhs(state, remaining_rule_text, name, rule_id, False)

    if remaining_rule_text and remaining_rule_text[0] == "\r":
        remaining_rule_text = (
            remaining_rule_text[2:]
            if remaining_rule_text[1] == "\n"
            else remaining_rule_text[1:]
        )
    elif remaining_rule_text and remaining_rule_text[0] == "\n":
        remaining_rule_text = remaining_rule_text[1:]
    elif remaining_rule_text:
        raise RuntimeError("expecting newline or end at " + remaining_rule_text)
    return remove_leading_white_space(remaining_rule_text, True)


def parse_ebnf(grammar_text: str) -> ParseState:
    try:
        state = ParseState()
        remaining_grammar_text = remove_leading_white_space(grammar_text, True)
        last_grammar_repr = ""
        while remaining_grammar_text:
            if last_grammar_repr:
                last_parsed_rule_len = len(last_grammar_repr) - len(
                    remaining_grammar_text
                )
                logger.debug(
                    f"last_parsed_rule: {last_grammar_repr[:last_parsed_rule_len]}"
                )
            last_grammar_repr = remaining_grammar_text
            remaining_grammar_text = parse_rule(state, remaining_grammar_text)
        return state
    except RuntimeError as err:
        logger.warning("error parsing grammar:", err)
        return ParseState()


###################################
# EBNF Grammar Parsing ends here  #
###################################


def break_grammar_into_rules(grammar_encoding: List[int]) -> List[List[int]]:
    offset = 0
    # we loop until we reach the end of the grammar_encoding
    rule_encodings = []
    i = 0
    while i < len(grammar_encoding) - 2:
        if (
            grammar_encoding[i] == END_OF_ALTERNATE_MARKER
            and grammar_encoding[i + 1] == END_OF_RULE_MARKER
        ):
            rule_encodings.append(grammar_encoding[offset : i + 2])
            offset = i + 2
            # skip the END_OF_RULE_MARKER
            # This is mandatory because if we do not skip the END_OF_RULE_MARKER
            # we fail in the case where the next rule has rule_id 0
            i += 1
        i += 1
    return rule_encodings


def break_rule_into_elements(rule_encoding: List[int]) -> List[List[int]]:
    rule_id = rule_encoding.pop(0)
    end_of_rule_marker = rule_encoding.pop(-1)
    assert (
        end_of_rule_marker == END_OF_RULE_MARKER
    ), f"rule should end with {END_OF_RULE_MARKER}, but got {end_of_rule_marker}"

    offset = 0
    elements = []
    while offset < len(rule_encoding):
        element_size = rule_encoding[offset]
        assert (
            rule_encoding[offset + element_size] == END_OF_ALTERNATE_MARKER
        ), f"element should end with {END_OF_ALTERNATE_MARKER}, but got {rule_encoding[offset + element_size]}"
        elements.append(rule_encoding[offset : offset + element_size + 1])
        offset += element_size + 1
    return elements


def _print_annotated_grammar(file, grammar_encoding, symbol_id_names, index=0):
    rule_id = grammar_encoding[index]
    print(f"<{index}>{symbol_id_names[rule_id]} ::=", end=" ", file=file)
    pos = index + 1
    while grammar_encoding[pos]:
        if pos - 1 > index:
            print("|", end=" ", file=file)
        pos += 1  # sequence size, not needed here
        while grammar_encoding[pos]:
            if grammar_encoding[pos] == REF_RULE_MARKER:
                ref_rule_id = grammar_encoding[pos + 1]
                print(
                    f"<{pos}>{symbol_id_names[ref_rule_id]}",
                    end=" ",
                    file=file,
                )
                pos += 2
            else:
                print("<{}>[".format(pos), end="", file=file)
                num_chars = grammar_encoding[pos]
                pos += 1

                for i in range(0, num_chars, 2):
                    print(
                        "{}-".format(chr(grammar_encoding[pos + i])), end="", file=file
                    )
                    if i + 1 < num_chars:
                        print(
                            "{}".format(chr(grammar_encoding[pos + i + 1])),
                            end="",
                            file=file,
                        )
                print("]", end=" ", file=file)
                pos += num_chars
        pos += 1
    print(file=file)
    return pos + 1


def print_grammar(file, state):
    pos = 0
    symbol_id_names = {v: k for k, v in state.symbol_table.items()}
    print("Grammar Rules:", file=file)
    while (
        pos < len(state.grammar_encoding)
        and state.grammar_encoding[pos] != END_OF_GRAMMAR_MARKER
    ):
        pos = _print_annotated_grammar(
            file, state.grammar_encoding, symbol_id_names, pos
        )
    if pos > len(state.grammar_encoding):
        raise Warning(f"grammar_encoding is not ended with {END_OF_GRAMMAR_MARKER}")
    pos = 0
    print("\nGrammar Hex representation:", file=file)
    while (
        pos < len(state.grammar_encoding)
        and state.grammar_encoding[pos] != END_OF_GRAMMAR_MARKER
    ):
        print(f"{state.grammar_encoding[pos]:04x}", end=" ", file=file)
        pos += 1
    if pos > len(state.grammar_encoding):
        raise Warning(f"grammar_encoding is not ended with {END_OF_GRAMMAR_MARKER}")
    else:
        print("ffff\n")

    print("Rules Decimal representation:", file=file)
    # we loop until we reach the end of the grammar_encoding
    rule_encodings = break_grammar_into_rules(state.grammar_encoding)
    for rule_encoding in rule_encodings:
        rule_id = rule_encoding[0]
        print(
            f"<{rule_id}> {break_rule_into_elements(rule_encoding)}",
            file=file,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse EBNF grammar files.")
    parser.add_argument(
        "-g",
        "--grammar-file",
        nargs="?",
        default="examples/grammars/json.ebnf",
        help="Path to the grammar file (default: examples/grammars/json.ebnf)",
    )

    args = parser.parse_args()

    with open(args.grammar_file, "r") as file:
        input_text = file.read()
    parsed_grammar = parse_ebnf(input_text)
    parsed_grammar.print()
    print(f"symbol_ids: \n{parsed_grammar.symbol_table}")

    start_rule_id = parsed_grammar.symbol_table["root"]
