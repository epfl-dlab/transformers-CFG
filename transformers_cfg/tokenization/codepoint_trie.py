#################
# DATA STRUCTURES
#################

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

LEAF = -1


def fmt_token_as_codepoints(token_id, tokenizer, only_ascii=True) -> List[int]:

    special_token_ids = tokenizer.additional_special_tokens_ids

    tokenizer_class_name = tokenizer.__class__.__name__.lower()

    if "gpt2" in tokenizer_class_name or "pretrained" in tokenizer_class_name:
        # GPT-2 or Pretrained tokenizers
        # No additional space handling needed
        handle_spaces = False
    elif "llama" in tokenizer_class_name or "t5" in tokenizer_class_name:
        # Llama or T5 tokenizers
        # Handle leading space in token
        handle_spaces = True
    else:
        # logger.warning(
        #     "Warning: unrecognized tokenizer: using default token formatting"
        # )
        handle_spaces = False

    if token_id in special_token_ids:
        return None
    token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    if handle_spaces:
        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        if raw_token.startswith("▁"):
            token = " " + token
    code_points = [ord(c) for c in token]
    # keep only code points within ASCII range
    code_points = code_points if all(c < 128 for c in code_points) else None
    return code_points


class CodePointTrie:
    def __init__(self, tokenizer, only_ascii=True):
        self.eos_token_id = tokenizer.eos_token_id
        self.all_token_codepoints = []
        self.trie = {}
        # we only keep ASCII code points
        # the reason why we should do this is because to handle unicode properly, we need to handle multi-byte characters
        # this can not be done with a simple code point trie
        # if we set only_ascii to False, we will be able to handle a subset of unicode characters
        # this behavior is probably not what we want
        self.only_ascii = only_ascii
        self.load_tokens(tokenizer)

    def id2str(self, token_id):
        return self.all_token_codepoints[token_id]

    def __len__(self):
        return len(self.all_token_codepoints)

    def load_tokens(self, tokenizer):
        self.all_token_codepoints = [
            fmt_token_as_codepoints(token_id, tokenizer, self.only_ascii)
            for token_id in range(len(tokenizer.get_vocab()))
        ]
        for token_id, token_codepoints in enumerate(self.all_token_codepoints):
            if token_codepoints is not None:
                self.insert_into_trie(self.trie, token_codepoints, token_id)

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
    token_trie = CodePointTrie(tokenizer)
