#################
# DATA STRUCTURES
#################


import re

LEAF = -1


def get_substitution(tokenizer):
    print(f"tokenizer type: {tokenizer.__class__.__name__}")
    if "gpt2" in tokenizer.__class__.__name__.lower():
        return BPESubstitution(tokenizer)
    elif (
        "llama" in tokenizer.__class__.__name__.lower()
        or "t5" in tokenizer.__class__.__name__.lower()
        or "bloom" in tokenizer.__class__.__name__.lower()
        or "phi" in tokenizer.__class__.__name__.lower()
    ):
        return Substitution(tokenizer)


class Substitution:
    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.tokenizer = tokenizer
        self.special = tokenizer.all_special_ids
        self.last_token_id = None

    def __len__(self):
        return len(self.tokenizer.get_vocab())

    def map(self, token_id: int) -> bytes:
        at_bos = False
        if self.last_token_id is not None and self.last_token_id == self.bos_token_id:
            at_bos = True
        self.last_token_id = token_id
        if token_id in self.special:
            return bytes()
        # if token_id is tensor, convert it to int
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        raw_token = self.tokenizer.convert_ids_to_tokens(token_id)
        # if the token is hex, token is a string like "<0x00>"
        # first 256 tokens are hex
        if raw_token.startswith("<0x"):
            hex_value = raw_token[4:-1]
            raw_token = chr(int(hex_value, 16))
        if raw_token.startswith("▁"):
            raw_token = raw_token.replace("▁", " ")
            if at_bos:
                # remove space at the beginning of the sentence
                raw_token = raw_token[1:]
        return bytes(raw_token, "utf-8")


class BPESubstitution(Substitution):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def map(self, token_id: int) -> bytes:
        # This is the case for BOS,
        # It should not be mapped to any token
        # if we decode it, it will be like <s>
        if token_id in self.special:
            return bytes()
        return bytes(
            self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
            "utf-8",
        )


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
            print("Warning: unrecognized tokenizer: using default token formatting")

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
