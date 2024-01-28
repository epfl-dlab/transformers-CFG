import logging
import sys
from typing import List

logger = logging.getLogger(__name__)

END_OF_ALTERNATE_MARKER = 0
END_OF_SIMPLE_RULE_MARKER = 0
END_OF_RULE_MARKER = 0
END_OF_GRAMMAR_MARKER = 0xFFFF
TO_BE_FILLED_MARKER = 0
REF_RULE_MARKER = 1
LITERAL_MARKER = 2


########################
# EBNF Grammar Parsing #
########################


class ParseState:
    def __init__(self):
        self.symbol_table = {}
        self.grammar_encoding = []  # old name: out_grammar
        self.grammar_encoding_rule_size = []

    def print(self, file=sys.stdout):
        pos = 0
        symbol_id_names = {v: k for k, v in self.symbol_table.items()}
        print("Grammar Rules:", file=file)

        while (
            pos < len(self.grammar_encoding)
            and self.grammar_encoding[pos] != END_OF_GRAMMAR_MARKER
        ):
            pos = print_rule(file, self.grammar_encoding, pos, symbol_id_names)
        if pos > len(self.grammar_encoding):
            raise Warning(f"grammar_encoding is not ended with {END_OF_GRAMMAR_MARKER}")
        pos = 0
        print("\nHex representation:", file=file)
        while (
            pos < len(self.grammar_encoding)
            and self.grammar_encoding[pos] != END_OF_GRAMMAR_MARKER
        ):
            print(f"{self.grammar_encoding[pos]:04x}", end=" ", file=file)
            pos += 1
        if pos > len(self.grammar_encoding):
            raise Warning(f"grammar_encoding is not ended with {END_OF_GRAMMAR_MARKER}")
        else:
            print("ffff\n")

        offset = 0
        print("Grammar Rule Sizes:", file=file)
        for i, rule_size in enumerate(self.grammar_encoding_rule_size):
            print(
                f"<{i}> {rule_size} {self.grammar_encoding[offset:offset + rule_size]}",
                file=file,
            )
            offset += rule_size


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


def parse_name(src) -> (str, str):
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


def parse_char(src) -> (str, str):
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
                    return (first << 4) + second, src[4:]
            raise RuntimeError("expecting \\xNN at " + src)
        elif esc in ('"', "[", "]"):
            return esc, src[2:]
        elif esc == "r":
            return "\r", src[2:]
        elif esc == "n":
            return "\n", src[2:]
        elif esc == "t":
            return "\t", src[2:]
        raise RuntimeError("unknown escape at " + src)
    elif src:
        return src[0], src[1:]
    raise RuntimeError("unexpected end of input")


def _parse_rhs_literal_string(src: str, outbuf: List[int]) -> str:
    assert src[0] == '"', f"rule should start with '\"', but got {src[0]}"
    remaining_src = src[1:]

    # advance until we get an end quote or run out of input
    while remaining_src and remaining_src[0] != '"':
        char, remaining_src = parse_char(remaining_src)
        outbuf.append(LITERAL_MARKER)
        outbuf.append(ord(char))
        outbuf.append(ord(char))

    # in case we ran out of input before finding the end quote
    if not remaining_src:
        raise RuntimeError(f"expecting an end quote at {src},but not found")

    # remove the end quote and return the remaining string
    return remaining_src[1:]


def _parse_rhs_char_ranges(src: str, outbuf: List[int]) -> str:
    assert src[0] == "[", f"rule should start with '[', but got {src[0]}"
    remaining_src = src[1:]
    start_idx = len(outbuf)
    # num chars in range - replaced at end of loop
    outbuf.append(TO_BE_FILLED_MARKER)
    while remaining_src and remaining_src[0] != "]":
        char, remaining_src = parse_char(remaining_src)

        outbuf.append(ord(char))
        if remaining_src[0] == "-" and remaining_src[1] != "]":
            endchar_pair, remaining_src = parse_char(remaining_src[1:])
            outbuf.append(ord(endchar_pair))
        else:
            # This is the case for enumerate, e.g., [0123456789], [abcdef]
            # Each char is considered as a range of itself, i.e., c-c
            outbuf.append(ord(char))
    if not remaining_src:
        raise RuntimeError(
            f"expecting an ] at {src},but not found, is the char range closed?"
        )
    # replace num chars with actual
    outbuf[start_idx] = len(outbuf) - start_idx - 1
    return remaining_src[1:]


def _parse_rhs_symbol_reference(src: str, state: ParseState, outbuf: List[int]) -> str:
    assert is_word_char(src[0]), f"rule should start with a word char, but got {src[0]}"
    name, remaining_src = parse_name(src)
    ref_rule_id = get_symbol_id(state, name)
    outbuf.append(REF_RULE_MARKER)
    outbuf.append(ref_rule_id)
    return remaining_src


def _parse_rhs_grouping(
    remaining_src: str, state: ParseState, rule_name: str, outbuf: List[int]
) -> str:
    assert (
        remaining_src[0] == "("
    ), f"rule should start with '(', but got {remaining_src[0]}"
    remaining_src = remove_leading_white_space(remaining_src[1:], True)
    # parse nested alternates into synthesized rule
    synthetic_rule_id = generate_symbol_id(state, rule_name)
    remaining_src = parse_rhs(state, remaining_src, rule_name, synthetic_rule_id, True)
    # output reference to synthesized rule
    outbuf.append(REF_RULE_MARKER)
    outbuf.append(synthetic_rule_id)

    if not remaining_src or remaining_src[0] != ")":
        raise RuntimeError("expecting ')' at " + remaining_src)
    return remaining_src[1:]


def _parse_rhs_repetition_operators(
    remaining_src: str,
    state: ParseState,
    rule_name: str,
    last_sym_start: int,
    outbuf: List[int],
) -> str:
    assert remaining_src[0] in (
        "*",
        "+",
        "?",
    ), f"rule should start with '*', '+', or '?', but got {remaining_src[0]}"
    out_grammar = state.grammar_encoding
    # last_sym_start = len(outbuf)

    # apply transformation to previous symbol (last_sym_start -
    # end) according to rewrite rules:
    # S* --> S' ::= S S' |
    # S+ --> S' ::= S S' | S
    # S? --> S' ::= S |
    sub_rule_id = generate_symbol_id(state, rule_name)
    out_grammar.append(sub_rule_id)
    sub_rule_start = len(out_grammar)
    # placeholder for size of 1st alternate
    out_grammar.append(TO_BE_FILLED_MARKER)
    # add preceding symbol to generated rule
    out_grammar.extend(outbuf[last_sym_start:])
    if remaining_src[0] in ("*", "+"):
        # cause generated rule to recurse
        out_grammar.append(REF_RULE_MARKER)
        out_grammar.append(sub_rule_id)
    # apply actual size
    out_grammar[sub_rule_start] = len(out_grammar) - sub_rule_start
    # mark end of 1st alternate
    out_grammar.append(END_OF_ALTERNATE_MARKER)
    sub_rule_start = len(out_grammar)
    # placeholder for size of 2nd alternate
    out_grammar.append(TO_BE_FILLED_MARKER)
    if remaining_src[0] == "+":
        # add preceding symbol as alternate only for '+'
        out_grammar.extend(outbuf[last_sym_start:])
    # apply actual size of 2nd alternate
    out_grammar[sub_rule_start] = len(out_grammar) - sub_rule_start
    # mark end of 2nd alternate, then end of rule
    out_grammar.append(END_OF_ALTERNATE_MARKER)
    out_grammar.append(END_OF_RULE_MARKER)

    # in original rule, replace previous symbol with reference to generated rule
    outbuf[last_sym_start:] = [REF_RULE_MARKER, sub_rule_id]
    return remaining_src[1:]


def parse_simple_rhs(state, rhs: str, rule_name: str, outbuf, is_nested):
    out_start_pos = len(outbuf)

    # sequence size, will be replaced at end when known
    outbuf.append(TO_BE_FILLED_MARKER)

    last_sym_start = len(outbuf)
    remaining_rhs = rhs
    while remaining_rhs:
        if remaining_rhs[0] == '"':  # literal string
            # mark the start of the last symbol, for repetition operator
            last_sym_start = len(outbuf)
            remaining_rhs = _parse_rhs_literal_string(remaining_rhs, outbuf)
        elif remaining_rhs[0] == "[":  # char range(s)
            # mark the start of the last symbol, for repetition operator
            last_sym_start = len(outbuf)
            remaining_rhs = _parse_rhs_char_ranges(remaining_rhs, outbuf)
        elif is_word_char(remaining_rhs[0]):  # rule reference
            # mark the start of the last symbol, for repetition operator
            last_sym_start = len(outbuf)
            remaining_rhs = _parse_rhs_symbol_reference(remaining_rhs, state, outbuf)
        elif remaining_rhs[0] == "(":  # grouping
            # mark the start of the last symbol, for repetition operator
            last_sym_start = len(outbuf)
            remaining_rhs = _parse_rhs_grouping(remaining_rhs, state, rule_name, outbuf)
        elif remaining_rhs[0] in ("*", "+", "?"):  # repetition operator
            # No need to mark the start of the last symbol, because we already did it
            if len(outbuf) - out_start_pos - 1 == 0:
                raise RuntimeError(
                    "expecting preceeding item to */+/? at " + remaining_rhs
                )
            remaining_rhs = _parse_rhs_repetition_operators(
                remaining_rhs, state, rule_name, last_sym_start, outbuf
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

    # apply actual size of this alternate sequence
    outbuf[out_start_pos] = len(outbuf) - out_start_pos
    # mark end of alternate
    outbuf.append(END_OF_SIMPLE_RULE_MARKER)
    return remaining_rhs


def parse_rhs(state, rhs: str, rule_name, rule_id, is_nested):
    outbuf = []
    remaining_rhs = parse_simple_rhs(state, rhs, rule_name, outbuf, is_nested)
    while remaining_rhs and remaining_rhs[0] == "|":
        remaining_rhs = remove_leading_white_space(remaining_rhs[1:], True)
        remaining_rhs = parse_simple_rhs(
            state, remaining_rhs, rule_name, outbuf, is_nested
        )

    # Now we have finished parsing the rhs, we can add the rule to the grammar_encoding
    state.grammar_encoding.append(rule_id)
    state.grammar_encoding.extend(outbuf)
    state.grammar_encoding.append(END_OF_RULE_MARKER)
    state.grammar_encoding_rule_size.append(len(outbuf) + 2)
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
        state.grammar_encoding.append(END_OF_GRAMMAR_MARKER)
        return state
    except RuntimeError as err:
        logger.warning("error parsing grammar:", err)
        return ParseState()


def print_rule(file, grammar_encoding, index, symbol_id_names):
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

    while state.grammar_encoding[pos] != END_OF_GRAMMAR_MARKER:
        pos = print_rule(file, state.grammar_encoding, pos, symbol_id_names)
    pos = 0
    print("\nHex representation:", file=file)
    while state.grammar_encoding[pos] != END_OF_GRAMMAR_MARKER:
        print(f"{state.grammar_encoding[pos]:04x}", end=" ", file=file)
        pos += 1
    print("ffff\n")

    offset = 0
    print("Grammar Rule Sizes:", file=file)
    for i, rule_size in enumerate(state.grammar_encoding_rule_size):
        print(
            f"<{i}> {rule_size} {state.grammar_encoding[offset:offset+rule_size]}",
            file=file,
        )
        offset += rule_size


###################################
# EBNF Grammar Parsing ends here  #
###################################
