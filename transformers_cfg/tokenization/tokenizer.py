import re
from typing import List


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

    def get_tokens_as_bytes(self) -> List[bytes]:
        vocab_size = self.real_vocab_size()
        token_as_bytes: List[bytes] = [
            self._format_token_as_bytes(i) for i in range(vocab_size)
        ]

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
