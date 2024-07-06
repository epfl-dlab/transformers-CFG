import re
from typing import List

from transformers_cfg.tokenization.middle.TokenizerMiddleMapping import (
    GPT2TokenizerMiddleMapping,
    LLAMA1TokenizerMiddleMapping,
)


def replace_hex(match):
    hex_value = match.group(1)
    return chr(int(hex_value, 16))


class Tokenizer:
    def __init__(self, hf_tokenizer):
        self.hf_tokenizer = hf_tokenizer
        self.special_token_ids = hf_tokenizer.all_special_ids

    def real_vocab_size(self):
        return len(self.hf_tokenizer.vocab)

    def getTokenizerMiddleMapping(self):
        raise NotImplementedError("This method should be implemented in the subclass")

    # def _format_gpt_token_as_bytes(self, token_id):
    #     if token_id in self.special:
    #         return None
    #     return bytes(
    #         self.hf_tokenizer.decode([id], clean_up_tokenization_spaces=False), "utf-8"
    #     )

    # def _format_llama_token_as_bytes(self, token_id):
    #     token = self.hf_tokenizer.convert_ids_to_tokens(token_id)
    #     token = re.sub(r"<0x([0-9a-fA-F]{2})>", replace_hex, token)
    #     token = token.replace("▁", " ")
    #     return bytes(token, "utf-8")

    # def _naive_format_token_as_bytes(self, token_id):
    #     token = self.hf_tokenizer.convert_ids_to_tokens(token_id)
    #     return bytes(token, "utf-8")

    def get_tokens_as_bytes(self) -> List[bytes]:

        # if (
        #     "gpt2" in self.hf_tokenizer.__class__.__name__.lower()
        #     or "bart" in self.hf_tokenizer.__class__.__name__.lower()
        #     or "pretrained" in self.hf_tokenizer.__class__.__name__.lower()
        # ):  # llama3 tokenizer
        #     special = self.hf_tokenizer.additional_special_tokens_ids

        #     # Here, the decoder does a string replace on a bunch of sequences
        #     # like ' .' for '.'. This interferes with our assumptions, where a
        #     # token should always have exactly one representation.
        #     # Fortunately(?) text-generation-inference doesn't seem to run this
        #     # cleanup, so we get extraneous spaces. So, in order to generate
        #     # the right token set for TGI, we have to skip the space trimming.
        #     # See:
        #     # https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3588-L3600
        #     def fmt_token(id):
        #         if id in special:
        #             return None
        #         return bytes(
        #             self.hf_tokenizer.decode([id], clean_up_tokenization_spaces=False), "utf-8"
        #         )

        # elif (
        #     "llama" in self.hf_tokenizer.__class__.__name__.lower()
        #     or "t5" in self.hf_tokenizer.__class__.__name__.lower()
        # ):

        #     def fmt_token(id):
        #         token = self.hf_tokenizer.convert_ids_to_tokens(id)
        #         token = re.sub(r"<0x([0-9a-fA-F]{2})>", replace_hex, token)
        #         token = token.replace("▁", " ")
        #         return bytes(token, "utf-8")

        # else:
        #     logger.warning(
        #         "Warning: unrecognized tokenizer: using default token formatting"
        #     )

        #     def fmt_token(id):
        #         token = tokenizer.convert_ids_to_tokens(id)
        #         return bytes(token, "utf-8")

        # note: vocab_size doesn't work here because there are also
        # get_added_vocab() tokens
        vocab_size = self.real_vocab_size()
        token_as_bytes: List[bytes] = [
            self._format_token_as_bytes(i) for i in range(vocab_size)
        ]
        # self.tokens = [fmt_token(i) for i in range(len(self.hf_tokenizer.get_vocab()))]
        # for token_id, token_bytes in enumerate(self.tokens):
        #     if token_bytes is not None:
        #         self.insert_into_trie(self.trie, token_bytes, token_id)

        return token_as_bytes

    @classmethod
    def from_hf_tokenizer(cls, hf_tokenizer):
        if (
            "gpt2" in hf_tokenizer.__class__.__name__.lower()
            or "bart" in hf_tokenizer.__class__.__name__.lower()
        ):
            return GPT2Tokenizer(hf_tokenizer)
        elif (
            "llama" in hf_tokenizer.__class__.__name__.lower()
            or "mistral" in hf_tokenizer.__class__.__name__.lower()
            or "t5" in hf_tokenizer.__class__.__name__.lower()
        ):
            return LlamaTokenizer(hf_tokenizer)
        elif "codegen" in hf_tokenizer.__class__.__name__.lower():
            # phi reuses the codegen tokenizer
            return PhiTokenizer(hf_tokenizer)
        else:
            raise NotImplementedError(
                f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}"
            )


class LlamaTokenizer(Tokenizer):
    def __init__(self, hf_tokenizer):
        super().__init__(hf_tokenizer)

    def _format_token_as_bytes(self, token_id):
        token = self.hf_tokenizer.convert_ids_to_tokens(token_id)
        token = re.sub(r"<0x([0-9a-fA-F]{2})>", replace_hex, token)
        token = token.replace("▁", " ")
        return bytes(token, "utf-8")


class GPT2Tokenizer(Tokenizer):
    def __init__(self, hf_tokenizer):
        super().__init__(hf_tokenizer)

    def _format_token_as_bytes(self, token_id):
        if token_id in self.special_token_ids:
            return None
        return bytes(
            self.hf_tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
            "utf-8",
        )


class CharacterTokenizer(Tokenizer):
    def __init__(self, hf_tokenizer):
        super().__init__(hf_tokenizer)

    def _format_token_as_bytes(self, token_id):
        return bytes(self.hf_tokenizer.convert_ids_to_tokens(token_id), "utf-8")


class PhiTokenizer(GPT2Tokenizer):
    def __init__(self, hf_tokenizer):
        super().__init__(hf_tokenizer)

    def real_vocab_size(self):
        return 50257  # 50 k tokens + 256 for bytes + 1 for EOS

    """


def getTokenizerMiddleMapping(tokenizer):

    if (
        "gpt2" in tokenizer.__class__.__name__.lower()
        or "codegen" in tokenizer.__class__.__name__.lower()
        or "bart" in tokenizer.__class__.__name__.lower()
    ):
        return GPT2TokenizerMiddleMapping(tokenizer)
    elif (
        "llama" in tokenizer.__class__.__name__.lower()
        or "mistral" in tokenizer.__class__.__name__.lower()
    ):
        return LLAMA1TokenizerMiddleMapping(tokenizer)
    elif "t5" in tokenizer.__class__.__name__.lower():
        return T5TokenizerMiddleMapping(tokenizer)
    else:
        raise NotImplementedError(f"Unicode mapping for {tokenizer.__class__.__name__}")

    """
