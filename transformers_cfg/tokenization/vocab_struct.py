#################
# DATA STRUCTURES
#################

import logging
import re

logger = logging.getLogger(__name__)

LEAF = -1


class TokenTrie:
    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
        self.tokens = []
        self.trie = {}
        self.load_tokens(tokenizer)

    def id2str(self, token_id):
        return self.tokens[token_id]

    def __len__(self):
        return len(self.tokens)

    def load_tokens(self, tokenizer):
        def replace_hex(match):
            hex_value = match.group(1)
            return chr(int(hex_value, 16))

        if "gpt2" in tokenizer.__class__.__name__.lower():
            special = tokenizer.additional_special_tokens_ids

            # Here, the decoder does a string replace on a bunch of sequences
            # like ' .' for '.'. This interferes with our assumptions, where a
            # token should always have exactly one representation.
            # Fortunately(?) text-generation-inference doesn't seem to run this
            # cleanup, so we get extraneous spaces. So, in order to generate
            # the right token set for TGI, we have to skip the space trimming.
            # See:
            # https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3588-L3600
            def fmt_token(id):
                if id in special:
                    return None
                return bytes(
                    tokenizer.decode([id], clean_up_tokenization_spaces=False), "utf-8"
                )

        elif (
            "llama" in tokenizer.__class__.__name__.lower()
            or "t5" in tokenizer.__class__.__name__.lower()
        ):

            def fmt_token(id):
                token = tokenizer.convert_ids_to_tokens(id)
                token = re.sub(r"<0x([0-9a-fA-F]{2})>", replace_hex, token)
                token = token.replace("▁", " ")
                return bytes(token, "utf-8")

        else:
            logger.warning(
                "Warning: unrecognized tokenizer: using default token formatting"
            )

            def fmt_token(id):
                token = tokenizer.convert_ids_to_tokens(id)
                return bytes(token, "utf-8")

        # note: vocab_size doesn't work here because there are also
        # get_added_vocab() tokens
        self.tokens = [fmt_token(i) for i in range(len(tokenizer.get_vocab()))]
        for token_id, token_bytes in enumerate(self.tokens):
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
