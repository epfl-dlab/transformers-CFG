import logging
import sys
import time
from abc import ABC
from functools import lru_cache
from typing import Dict, List

import torch

from .vocab_struct import LEAF, TokenTrie

logger = logging.getLogger(__name__)


########################
# EBNF Grammar Parsing #
########################

END_OF_ALTERNATE_MARKER = 0
END_OF_RULE_MARKER = 0
TO_BE_FILLED_MARKER = 0
REF_RULE_MARKER = 1
LITERAL_MARKER = 2


class ParseState:
    def __init__(self):
        self.symbol_ids = {}
        self.grammar_encoding = []  # old name: out_grammar
        self.grammar_encoding_rule_size = []


def get_symbol_id(state, src):
    if src not in state.symbol_ids:
        state.symbol_ids[src] = len(state.symbol_ids)
    return state.symbol_ids[src]


def generate_symbol_id(state, base_name):
    next_id = len(state.symbol_ids)
    state.symbol_ids[base_name + "_" + str(next_id)] = next_id
    return next_id


def is_word_char(c):
    return c.isalnum() or c == "-" or c == "_"


def hex_to_int(c):
    if c.isdigit():
        return int(c)
    elif "a" <= c.lower() <= "f":
        return ord(c.lower()) - ord("a") + 10
    return -1


def remove_leading_white_space(src, newline_ok):
    """
    Skips over whitespace and comments in the input string.

    This function processes the input string, skipping over any spaces, tabs,
    and content following a '#' character, which denotes a comment. The parsing
    of a comment continues until the end of the line (denoted by newline characters
    '\r' or '\n'). If the 'newline_ok' parameter is set to False, the function
    will stop processing and return the remaining string upon encountering a
    newline character, otherwise it will skip over newline characters as well.

    Parameters:
    src (str): The input string to be processed.
    newline_ok (bool): A flag indicating whether encountering a newline character
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
            if not newline_ok and src[pos] in ("\r", "\n"):
                break
            pos += 1
    return src[pos:]


def parse_name(src):
    pos = 0
    while pos < len(src) and is_word_char(src[pos]):
        pos += 1
    if pos == 0:
        raise RuntimeError("expecting name at " + src)
    return src[:pos], src[pos:]


def parse_char(src):
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


def parse_sequence(state, src, rule_name, outbuf, is_nested):
    out_start_pos = len(outbuf)

    # sequence size, will be replaced at end when known
    outbuf.append(TO_BE_FILLED_MARKER)

    last_sym_start = len(outbuf)
    remaining_src = src
    while remaining_src:
        if remaining_src[0] == '"':  # literal string
            remaining_src = remaining_src[1:]
            last_sym_start = len(outbuf)
            while remaining_src[0] != '"':
                char, remaining_src = parse_char(remaining_src)

                # each char of a literal is encoded as a "range" of char - char
                outbuf.append(LITERAL_MARKER)
                outbuf.append(ord(char))
                outbuf.append(ord(char))
            remaining_src = remove_leading_white_space(remaining_src[1:], is_nested)
        elif remaining_src[0] == "[":  # char range(s)
            remaining_src = remaining_src[1:]
            last_sym_start = len(outbuf)
            # num chars in range - replaced at end of loop
            outbuf.append(TO_BE_FILLED_MARKER)
            while remaining_src[0] != "]":
                char, remaining_src = parse_char(remaining_src)

                outbuf.append(ord(char))
                if remaining_src[0] == "-" and remaining_src[1] != "]":
                    endchar_pair, remaining_src = parse_char(remaining_src[1:])
                    outbuf.append(ord(endchar_pair))
                else:
                    # chars that aren't part of a c1-c2 range are just doubled (i.e., c-c)
                    outbuf.append(ord(char))
            # replace num chars with actual
            outbuf[last_sym_start] = len(outbuf) - last_sym_start - 1
            remaining_src = remove_leading_white_space(remaining_src[1:], is_nested)
        elif is_word_char(remaining_src[0]):  # rule reference
            name, remaining_src = parse_name(remaining_src)
            ref_rule_id = get_symbol_id(state, name)
            remaining_src = remove_leading_white_space(remaining_src, is_nested)
            last_sym_start = len(outbuf)
            outbuf.append(REF_RULE_MARKER)
            outbuf.append(ref_rule_id)
        elif remaining_src[0] == "(":  # grouping
            # parse nested alternates into synthesized rule
            remaining_src = remove_leading_white_space(remaining_src[1:], True)
            sub_rule_id = generate_symbol_id(state, rule_name)
            remaining_src = parse_alternates(
                state, remaining_src, rule_name, sub_rule_id, True
            )
            last_sym_start = len(outbuf)
            # output reference to synthesized rule
            outbuf.append(REF_RULE_MARKER)
            outbuf.append(sub_rule_id)
            if remaining_src[0] != ")":
                raise RuntimeError("expecting ')' at " + remaining_src)
            remaining_src = remove_leading_white_space(remaining_src[1:], is_nested)
        elif remaining_src[0] in ("*", "+", "?"):  # repetition operator
            if len(outbuf) - out_start_pos - 1 == 0:
                raise RuntimeError(
                    "expecting preceeding item to */+/? at " + remaining_src
                )
            out_grammar = state.grammar_encoding

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
            outbuf[last_sym_start:] = [1, sub_rule_id]

            remaining_src = remove_leading_white_space(remaining_src[1:], is_nested)
        else:
            break
    # apply actual size of this alternate sequence
    outbuf[out_start_pos] = len(outbuf) - out_start_pos
    # mark end of alternate
    outbuf.append(END_OF_ALTERNATE_MARKER)
    return remaining_src


def parse_alternates(state, src, rule_name, rule_id, is_nested):
    outbuf = []
    remaining_src = parse_sequence(state, src, rule_name, outbuf, is_nested)
    while remaining_src and remaining_src[0] == "|":
        remaining_src = remove_leading_white_space(remaining_src[1:], True)
        remaining_src = parse_sequence(
            state, remaining_src, rule_name, outbuf, is_nested
        )

    state.grammar_encoding.append(rule_id)
    state.grammar_encoding.extend(outbuf)
    state.grammar_encoding.append(0)
    state.grammar_encoding_rule_size.append(len(outbuf) + 2)
    return remaining_src


def parse_rule(state, src):
    name, remaining_src = parse_name(src)
    remaining_src = remove_leading_white_space(remaining_src, False)
    rule_id = get_symbol_id(state, name)

    if remaining_src[:3] != "::=":
        raise RuntimeError("expecting ::= at " + remaining_src)
    remaining_src = remove_leading_white_space(remaining_src[3:], True)

    remaining_src = parse_alternates(state, remaining_src, name, rule_id, False)

    if remaining_src and remaining_src[0] == "\r":
        remaining_src = (
            remaining_src[2:] if remaining_src[1] == "\n" else remaining_src[1:]
        )
    elif remaining_src and remaining_src[0] == "\n":
        remaining_src = remaining_src[1:]
    elif remaining_src:
        raise RuntimeError("expecting newline or end at " + remaining_src)
    return remove_leading_white_space(remaining_src, True)


def parse_ebnf(src):
    try:
        state = ParseState()
        grammar_repr = remove_leading_white_space(src, True)
        last_grammar_repr = ""
        while grammar_repr:
            if last_grammar_repr:
                last_parsed_rule_len = len(last_grammar_repr) - len(grammar_repr)
                logger.debug(
                    f"last_parsed_rule: {last_grammar_repr[:last_parsed_rule_len]}"
                )
            last_grammar_repr = grammar_repr
            grammar_repr = parse_rule(state, grammar_repr)
        state.grammar_encoding.append(0xFFFF)
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
    symbol_id_names = {v: k for k, v in state.symbol_ids.items()}
    print("Grammar Rules:", file=file)

    while state.grammar_encoding[pos] != 0xFFFF:
        pos = print_rule(file, state.grammar_encoding, pos, symbol_id_names)
    pos = 0
    print("\nBinary representation:", file=file)
    while state.grammar_encoding[pos] != 0xFFFF:
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


class AbstractGrammarConstraint(ABC):
    def __init__(self, grammar_str, start_rule_name, tokenizer):
        state = parse_ebnf(grammar_str)
        grammar_encoding = state.grammar_encoding
        self.start_rule_id = state.symbol_ids.get(start_rule_name)

        self.eos_token_id = tokenizer.eos_token_id
        self.token_trie = TokenTrie(tokenizer)
        self.tokenizer = tokenizer
        self.grammar_encoding = grammar_encoding

        pos = 0
        rules: Dict[int, int] = {}

        while grammar_encoding[pos] != 0xFFFF:
            rule_id = grammar_encoding[pos]

            # Store the current position in the 'rules' list at the index corresponding to rule_id.
            # This effectively maps each rule_id to its position in the grammar encoding.
            rules[rule_id] = pos
            pos += 1

            # Continue to the next rule in the encoding.
            # The loop advances by the size indicated at the current position (grammar_encoding[pos])
            # plus one for the size field itself.
            while grammar_encoding[pos]:
                pos += 1 + grammar_encoding[pos]
            # Now we're at the end of the rule,
            # so advance to the next rule by skipping the 0, which means 'end of rule'.
            pos += 1

        self.start_rule_pos = rules[self.start_rule_id]
        self.rules_pos_dict: Dict[int, int] = rules

    def init_stacks(self):
        # suppose the start rule position is 0, then grammar_encoding[0] = rule_id
        # grammar_encoding[1] = rule_size
        # grammar_encoding[2] = rule_type
        # this is why we need to add 2 to the start rule position
        stack = [self.start_rule_pos + 2]
        # convert to tuple for caching(immutable)
        return self.advance_stack(tuple(stack))

    # For each stack, resolve rules to find the actual characters that are
    # accepted by this stack (not the set of sub-rules).
    # This is where the parsing happens.
    # The parsing is a top-down, left-to-right, depth-first traversal of the
    # grammar.
    @lru_cache(maxsize=32768)
    def advance_stack(self, stack):
        stack = list(stack)
        # If the stack is empty, we're done. Because no more tokens should be accepted.
        if len(stack) == 0:
            return [stack]

        # Get the top of the stack.
        pos = stack[-1]

        # If the stack head is a terminal(literal), we can resolve it immediately.
        # literal is marked with 2 in the grammar encoding.
        if self.grammar_encoding[pos] > 1:
            return [stack]

        # The stack head is a nonterminal (a rule reference, 1 in the grammar encoding).
        # Resolving this rule gives a set of one or more possible positions
        # (e.g. two in `a ::= b | c`)
        # We pop the current rule off the stack and, for each option, push:
        # - the symbol following this symbol in the current rule; then
        # - the first symbol of the resolved rule.
        referenced_rule_id = self.grammar_encoding[pos + 1]

        # subpos should points to the size of the subrule
        subpos = self.rules_pos_dict[referenced_rule_id] + 1
        stacks: List[List[int]] = []

        # do depth-first search to find all possible rules and check the next terminal
        # When this value is non-zero, it indicates that subpos is not yet at the end of the rule, so we can continue.
        # here subpos is a pointer, and the value in the rule encoding can never be 0 except for the end of the rule.
        while self.grammar_encoding[subpos]:
            new_stack = stack[:-1]
            if self.grammar_encoding[pos + 2]:
                # check if there is a next symbol in the current rule, e.g. `a ::= b c | d`
                # if yes, push the pos to rule_size to the stack
                new_stack.append(pos + 2)

            # if the type of the next symbol is not "empty", push the first symbol of the resolved rule to the stack
            if self.grammar_encoding[subpos + 1]:
                new_stack.append(subpos + 1)
            stacks.extend(self.advance_stack(tuple(new_stack)))
            # The increment subpos += self.grammar_encoding[subpos] + 1
            # moves subpos forward in the grammar encoding array to the next alternative in the current rule.
            subpos += self.grammar_encoding[subpos] + 1
        return stacks

    def _consume_char(self, byte, stacks):
        new_stacks = []
        for stack in stacks:
            # stack is empty
            if not stack:
                continue

            pos = stack[-1]
            num_chars = self.grammar_encoding[pos]

            # to make pos point to the size of the char range rule
            pos += 1
            found = False
            for i in range(0, num_chars, 2):
                if (
                    self.grammar_encoding[pos + i] <= byte
                    and byte <= self.grammar_encoding[pos + i + 1]
                ):
                    found = True
                    break
            if not found:
                continue

            pos += num_chars
            new_stack = stack[:-1]
            if self.grammar_encoding[pos]:
                new_stack.append(pos)
            new_stacks.extend(self.advance_stack(tuple(new_stack)))

        return new_stacks

    def _consume_string(self, string: str, stacks: List[List[int]]):
        _bytes = bytes(string, "utf-8")
        for byte in _bytes:
            stacks = self._consume_char(byte, stacks)
        return stacks

    def _consume_token_id(self, token_id: int, stacks: List[List[int]]):
        if token_id == self.eos_token_id:
            if stacks and all(len(stack) != 0 for stack in stacks):
                raise Exception(
                    f"At least one of the stack should be empty when EOS is reached. However, "
                    f"the stacks are {stacks}"
                )
            return []

        for byte in self.token_trie.id2str(token_id):
            stacks = self._consume_char(byte, stacks)
            # check updated stacks
            # TODO, I commented this out because it will fail when the stack is empty
            # empty stack means the end of the grammar
            # assert stacks != []

        return stacks

    def advance_token_ids(self, *args, **kwargs):
        """Process a list of tokens according to the grammar rules."""
        raise NotImplementedError

    def batch_filter_vocab(self, batch_stacks, device) -> torch.Tensor:
        batch_acceptance = []
        for stacks in batch_stacks:
            batch_acceptance.append(self.filter_vocab(stacks, device))
        return torch.stack(batch_acceptance)

    def filter_vocab(self, stacks, device) -> torch.Tensor:
        if not stacks:  # Check if stacks is empty
            # Handle the empty case: for example, return a tensor of False
            # The size of the tensor should match the size of your vocabulary
            vocab_size = len(self.token_trie)
            logger.debug(f"Empty stack, sum of acceptance: {0}")
            return torch.zeros(vocab_size, dtype=torch.bool, device=device)

        acceptance_matrix = torch.cat(
            [self.token_acceptance_for_stack(tuple(stack), device) for stack in stacks]
        )
        # Merge stacks: any True => True
        acceptance = acceptance_matrix.reshape(len(stacks), -1).any(dim=0)
        logger.debug(f"sum of acceptance: {acceptance.sum()}")
        return acceptance

    # For each sub-rule in the grammar, cache whether each byte is accepted.
    @lru_cache(maxsize=None)
    def char_acceptance_at_rule_pos(self, rule_pos):
        # every time this function is called, the result is cached
        # next time when the same pos is called, the result is returned directly
        # Here the pos corresponds to the literal or char range rule
        # it doesn't handle the rule reference
        acceptance = [False] * 256
        num_chars = self.grammar_encoding[rule_pos]
        rule_pos += 1
        for i in range(0, num_chars, 2):
            start = self.grammar_encoding[rule_pos + i]
            end = self.grammar_encoding[rule_pos + i + 1]
            for j in range(start, end + 1):
                acceptance[j] = True
        return acceptance

    # Probably this should be configurable. If the grammar has an exceedingly
    # large number of states, the correct setting is a tradeoff between GPU
    # RAM usage and recomputation time.
    #
    # The main variable that pushes usage up here is number of states in the
    # grammar.
    @lru_cache(maxsize=32768)
    def token_acceptance_for_stack(self, stack, device):
        stack = list(stack)  # needs to come in as a tuple for lru_cache

        accepts = [False] * len(self.token_trie)
        accepts[self.eos_token_id] = len(stack) == 0
        if len(stack) == 0:
            logger.debug("empty stack")

        def traverse_trie(trie, stacks):
            for byte, next_trie in trie.items():
                if byte == LEAF:
                    token_id = next_trie
                    if token_id != self.eos_token_id:
                        # if the stacks is not empty, it means we can still continue to parse
                        # so we should accept the token
                        accepts[token_id] = bool(stacks)
                    continue

                new_stacks = []
                for stk in stacks:
                    if not stk:
                        continue

                    next_rule_pos = stk[-1]
                    num_chars = self.grammar_encoding[next_rule_pos]

                    if not self.char_acceptance_at_rule_pos(next_rule_pos)[byte]:
                        # if the current byte is not accepted by the current rule, we need to try next rule
                        continue

                    next_rule_pos += num_chars + 1
                    new_stack = stk[:-1]
                    if self.grammar_encoding[next_rule_pos]:
                        new_stack.append(next_rule_pos)
                    new_stacks.extend(self.advance_stack(tuple(new_stack)))

                if new_stacks:
                    traverse_trie(next_trie, new_stacks)

        traverse_trie(self.token_trie.trie, [stack])

        x = torch.tensor(accepts, dtype=torch.bool, device=device)
        return x


class IncrementalGrammarConstraint(AbstractGrammarConstraint):
    def __init__(self, grammar_str, start_rule_name, tokenizer):
        super().__init__(grammar_str, start_rule_name, tokenizer)
        self.last_size = None

        # if self.last_size is not set (which would be the case when processing the first token).
        # In this case, do nothing.

    def advance_token_ids(self, input_ids, batch_stacks, parse_start_index=None):

        if self.last_size is None:
            prefix_to_parse = [
                single_input_ids[parse_start_index:]
                if parse_start_index is not None
                else []
                for single_input_ids in input_ids
            ]

            # self.grammar_acceptor.accept_token_ids(prefix_to_parse, self.stacks)
            batch_stacks = [
                self._consume_token_ids(prefix, stack)
                for prefix, stack in zip(prefix_to_parse, batch_stacks)
            ]
            #  if the length of the current input IDs (input_ids[0]) is exactly one more than self.last_size.
            #  This is expected in a scenario where inputs are processed incrementally, one token at a time.
        elif len(input_ids[0]) == self.last_size + 1:
            batch_stacks = [
                self._consume_token_id(single_input_ids[-1], stack)
                for single_input_ids, stack in zip(input_ids, batch_stacks)
            ]
            #  ensure that the input size is consistent with the expected incremental processing
            #  (i.e., one token at a time).
        else:
            # here we check if the input_ids are one token longer than the last time we processed
            # but we don't check if input_ids are actually valid.
            # Imagine a scenario where we generate 10 tokens, then we replace the 10 generated tokens with 10 new tokens.
            # In this case, the input_ids will be consistent with the last_size, but the input_ids are not valid.
            # However, should we really check if the input_ids are valid here?
            # If we do, then we need to reparse the whole input_ids at each call, which is not efficient.
            # Maybe we should just trust the user to provide valid input_ids?
            # The conclusion is that, we assume the input_ids are valid, and our generation will be correct.
            # If the input_ids are not valid, then the generation result will be wrong and we don't take responsibility for that.
            raise RuntimeError(
                "Input ID's length is inconsistent with the current state of "
                "the GrammarConstrainedLogitsProcessor. If you want to process "
                "another input sequence, please instantiate a new "
                "GrammarConstrainedLogitsProcessor."
            )
        self.last_size = len(input_ids[0])

        return batch_stacks

    def _consume_token_ids(
        self, token_ids: List[int], stacks: List[List[int]], as_string=True
    ):
        if as_string:
            string = self.tokenizer.decode(token_ids)
            stacks = self._consume_string(string, stacks)
        else:
            for token_id in token_ids:
                stacks = self._consume_token_id(token_id, stacks)
        return stacks


class VanillaGrammarConstraint(AbstractGrammarConstraint):
    def __init__(self, grammar_str, start_rule_name, tokenizer):
        super().__init__(grammar_str, start_rule_name, tokenizer)
        self.offset = None

    def advance_token_ids(self, input_ids, batch_stacks, parse_start_index=None):
        # By design, the batch_stacks should be empty at the beginning, thus it doesn't matter what we pass in.
        if self.offset is None:
            self.offset = (
                len(input_ids[0]) if parse_start_index is None else parse_start_index
            )

        batch_stacks_from_scratch = [self.init_stacks() for _ in range(len(input_ids))]

        prefix_to_consume = [
            single_input_ids[self.offset :] for single_input_ids in input_ids
        ]
        # self.grammar_acceptor.accept_token_ids(prefix_to_consume, self.stacks)
        advanced_batch_stacks = [
            self._consume_token_ids(prefix, stack, as_string=False)
            for prefix, stack in zip(prefix_to_consume, batch_stacks_from_scratch)
        ]
        return advanced_batch_stacks

    def _consume_token_ids(
        self, token_ids: List[int], stacks: List[List[int]], as_string=True
    ):
        if as_string:
            string = self.tokenizer.decode(token_ids)
            stacks = self._consume_string(string, stacks)
        else:
            for token_id in token_ids:
                stacks = self._consume_token_id(token_id, stacks)
        return stacks


if __name__ == "__main__":
    # set logging level
    logging.basicConfig(level=logging.DEBUG)

    try:
        with open("examples/grammars/grammar_t5_fail_1.ebnf", "r") as file:
            input_text = file.read()
        state = parse_ebnf(input_text)
        print_grammar(sys.stdout, state)
        print(f"symbol_ids: \n{state.symbol_ids}")

    except FileNotFoundError:
        print("Error: File 'grammar.ebnf' not found.")
    except IOError as e:
        print("Error reading file 'grammar.ebnf':", e)
