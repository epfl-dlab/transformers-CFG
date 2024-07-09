import re
from typing import List
from transformers import (
    GPT2TokenizerFast,
    BartTokenizerFast,
    LlamaTokenizerFast,
    T5TokenizerFast,
    CodeGenTokenizerFast,
    PreTrainedTokenizerFast,
)

from transformers_cfg.tokenization.SUPPORTED_TOKENIZERS import SUPPORTED_TOKENIZERS
from transformers_cfg.tokenization.utils import (
    replace_hex,
)


def get_TCFG_tokenizer_class(model_name_or_tokenizer):
    from transformers import AutoTokenizer

    if isinstance(model_name_or_tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_tokenizer)
    else:
        tokenizer = model_name_or_tokenizer

    return TCFG_Tokenizer.from_hf_tokenizer(tokenizer).__class__


class TCFG_Tokenizer:
    def __init__(self, hf_tokenizer):
        self.hf_tokenizer = hf_tokenizer
        self.special_token_ids = hf_tokenizer.all_special_ids

    def real_vocab_size(self):
        return len(self.hf_tokenizer.vocab)

    def get_tokens_as_bytes(self) -> List[bytes]:
        vocab_size = self.real_vocab_size()
        token_as_bytes: List[bytes] = [
            self._format_token_as_bytes(i) for i in range(vocab_size)
        ]

        return token_as_bytes

    @classmethod
    def from_hf_tokenizer(cls, hf_tokenizer):
        assert (
            type(hf_tokenizer) in SUPPORTED_TOKENIZERS
        ), f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}, supported tokenizers: {SUPPORTED_TOKENIZERS}"

        if isinstance(
            hf_tokenizer,
            (GPT2TokenizerFast, BartTokenizerFast),
        ):
            return TCFG_GPT2Tokenizer(hf_tokenizer)
        elif isinstance(hf_tokenizer, (LlamaTokenizerFast, T5TokenizerFast)):
            return TCFG_LlamaTokenizer(hf_tokenizer)
        elif isinstance(hf_tokenizer, CodeGenTokenizerFast):
            # phi reuses the codegen tokenizer
            return TCFG_PhiTokenizer(hf_tokenizer)
        elif isinstance(
            hf_tokenizer, PreTrainedTokenizerFast
        ) and hf_tokenizer.name_or_path.startswith("meta-llama/Meta-Llama-3"):
            return TCFG_LlamaTokenizer(hf_tokenizer)
        else:
            raise NotImplementedError(
                f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}"
            )


class TCFG_LlamaTokenizer(TCFG_Tokenizer):
    def __init__(self, hf_tokenizer):
        super().__init__(hf_tokenizer)

    def _format_token_as_bytes(self, token_id):
        token = self.hf_tokenizer.convert_ids_to_tokens(token_id)
        token = re.sub(r"<0x([0-9a-fA-F]{2})>", replace_hex, token)
        # token = token.replace("▁", " ")
        return bytes(token, "utf-8")


class TCFG_GPT2Tokenizer(TCFG_Tokenizer):
    def __init__(self, hf_tokenizer):
        super().__init__(hf_tokenizer)

    def _format_token_as_bytes(self, token_id):
        if token_id in self.special_token_ids:
            return None
        return bytes(
            self.hf_tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
            "utf-8",
        )


class TCFG_CharacterTokenizer(TCFG_Tokenizer):
    """
    Not yet used, but can be used for character level tokenization (even though rarely used in practice)
    """

    def __init__(self, hf_tokenizer):
        super().__init__(hf_tokenizer)

    def _format_token_as_bytes(self, token_id):
        return bytes(self.hf_tokenizer.convert_ids_to_tokens(token_id), "utf-8")


class TCFG_PhiTokenizer(TCFG_GPT2Tokenizer):
    def __init__(self, hf_tokenizer):
        super().__init__(hf_tokenizer)

    def real_vocab_size(self):
        return 50257  # 50 k tokens + 256 for bytes + 1 for EOS
