import logging
from functools import lru_cache
from typing import Dict, List, Tuple
from collections import deque

from transformers_cfg.mapping import get_mapping

# from transformers_cfg.parser import parse_ebnf
# from transformers_cfg.recognizer import GrammarRecognizer
# from transformers_cfg.token_grammar_recognizer import IncrementalTokenGrammarRecognizer

logger = logging.getLogger(__name__)

# def check_token_acceptance_in_trie(trie, stacks, grammar, partial_utf8, accept_eos=True, eos_token_id=None) -> List[bool]:
#     accept_f = lambda x: grammar._probe_bytes_partial_match(x, stack=stacks, partial_utf8=partial_utf8)
#     accepts = trie.get_token_acceptance(accept=accept_f, accept_eos=accept_eos, eos_token_id=eos_token_id)
#     return accepts


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.token_id = None


class ByteTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, token_id=None):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.token_id = token_id

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def start_with_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    @classmethod
    def from_tokenizer(cls, tokenizer, unicode=True):
        vocab: Dict[str, int] = tokenizer.get_vocab()
        trie = cls()
        mapping = get_mapping(tokenizer, unicode=unicode)
        for token_id in vocab.values():
            byte_repr = mapping.map(token_id)
            trie.insert(byte_repr, token_id)
        return trie

    @lru_cache(maxsize=128)
    def __len__(self):
        return len(self.dfs(verbose=False))

    def dfs(self, accept=lambda x: True, verbose=False) -> List[Tuple[List[int], int]]:
        result = []
        counter = {"visited": 0, "pruned": 0}
        _dfs(self.root, [], result, accept, counter)
        return result

    def bfs(
        self, predicate=lambda x: True, verbose=False
    ) -> List[Tuple[List[int], int]]:
        queue = deque([(self.root, [])])
        valid_byte_seqs: List[Tuple[List[int], int]] = []
        counter = {"visited": 0, "pruned": 0}

        while queue:
            counter["visited"] += 1
            node, byte_seq = queue.popleft()
            if predicate(byte_seq):
                if node.is_end_of_word:
                    valid_byte_seqs.append((byte_seq, node.token_id))
                for char, next_node in node.children.items():
                    new_byte_seq: List[int] = byte_seq.copy()
                    new_byte_seq.append(char)
                    queue.append((next_node, new_byte_seq))
            else:
                counter["pruned"] += 1
        return valid_byte_seqs

    def get_token_acceptance(
        self, accept=lambda x: True, accept_eos=True, eos_token_id=None
    ) -> List[bool]:
        valid_byte_seqs: List[Tuple[List[int], int]] = self.bfs(accept, verbose=True)
        valid_token_ids: List[int] = [token_id for _, token_id in valid_byte_seqs]
        token_acceptance: List[bool] = [False] * (len(self))
        for token_id in valid_token_ids:
            token_acceptance[token_id] = True
        if not accept_eos:
            # eos_token is mapped to an empty string, so it's always accepted regardless of the accept function
            # this can be undesirable, so we can set it to False to ignore it
            token_acceptance[eos_token_id] = False
        return token_acceptance


def _dfs(
    node,
    cur_byte_seq: List[int],
    result: List[Tuple[List[int], int]],
    accept: callable,
    counter: Dict[str, int],
):
    counter["visited"] += 1
    if accept(cur_byte_seq):
        if node.is_end_of_word:
            result.append((cur_byte_seq, node.token_id))
        for char, next_node in node.children.items():
            new_byte_seq: List[int] = cur_byte_seq.copy()
            new_byte_seq.append(char)
            _dfs(next_node, new_byte_seq, result, accept, counter)
    else:
        # Skip the entire subtree if the predict function returns False
        counter["pruned"] += 1
        return


def starts_with_prefix(prefix, target):
    """
    Check if the given prefix is a valid start of the target word or if the target word is a valid start of the given prefix.

    Args:
    prefix (str): The string prefix to be checked.
    target (str): The target word to compare the prefix against.

    Returns:
    bool: True if prefix is a valid start of target or if target is a valid start of prefix, False otherwise.
    """

    # Check if the target word starts with the given prefix.
    # This covers the case where the prefix is shorter than the target word.
    if target.startswith(prefix):
        return True

    # Check if the given prefix starts with the target word.
    # This covers the case where the prefix is longer than or equal to the target word.
    if prefix.startswith(target):
        return True

    # If neither of the above conditions are true, return False.
    return False


if __name__ == "__main__":
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=True)

    trie = ByteTrie.from_tokenizer(tokenizer, unicode=True)
    print(f"length of trie: {len(trie)}=={len(tokenizer.vocab.items())}")

    #
    # print(trie.search("hello"))  # Example, replace with actual words from the vocab
    # print(trie.start_with_prefix("hell"))
    #
    # # Example Usage
    # words = trie.dfs(accept=lambda x: len(x) > 0 and x[0] == 65 or len(x)==0)
    # for word in words:
    #     print(bytes(word[0]).decode("utf-8"))
    #
    # # Example Usage
    # words = trie.bfs(predicate=lambda x: len(x) > 0 and x[0] == 65 or len(x)==0)
    # for word in words:
    #     print(bytes(word[0]).decode("utf-8"))
    #
    # token_acceptance = trie.get_token_acceptance(accept=lambda x: len(x) > 0 and x[0] == 65 or len(x)==0)
    # print(sum(token_acceptance))
    # assert sum(token_acceptance) == len(words)

    ########################
    # UTF-8
    ########################

    # from transformers import AutoTokenizer
    #
    # japanese = "こんにちは世界"
    # with open("examples/grammars/japanese.ebnf", "r") as file:
    #     input_text = file.read()
    # parsed_grammar = parse_ebnf(input_text)
    #
    # start_rule_id = parsed_grammar.symbol_table["root"]
    #
    # recognizer = GrammarRecognizer(parsed_grammar.grammar_encoding, start_rule_id)
    # accept_state = recognizer.init_accept_state()
    # token_acc = trie.get_token_acceptance(accept=lambda x: recognizer._probe_bytes_partial_match(x, accept_state=accept_state))
