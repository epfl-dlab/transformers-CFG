#################
# DATA STRUCTURES
#################

import logging
from typing import List

from transformers_cfg.tokenization.tokenizer import TCFG_Tokenizer

logger = logging.getLogger(__name__)

LEAF = -1


class TokenTrie:
    # Not used anymore, replaced by ByteTrie
    def __init__(self, hf_tokenizer):
        self.eos_token_id = hf_tokenizer.eos_token_id
        self.tcfg_tokenizer = TCFG_Tokenizer.from_hf_tokenizer(hf_tokenizer)
        self.trie = {}

        self.load_token_as_bytes()

    def id2str(self, token_id):
        return self.tokens[token_id]

    def __len__(self):
        return self.tcfg_tokenizer.real_vocab_size()

    def load_token_as_bytes(self):
        tokens: List[bytes] = self.tcfg_tokenizer.get_tokens_as_bytes()
        for token_id, token_bytes in enumerate(tokens):
            if token_bytes is not None:
                self.insert_into_trie(self.trie, token_bytes, token_id)

    def insert_into_trie(self, trie, token_bytes, token_id):
        current = trie
        for byte in token_bytes:
            if byte not in current:
                current[byte] = {}
            current = current[byte]
        current[LEAF] = token_id


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    token_trie = TokenTrie(tokenizer)
