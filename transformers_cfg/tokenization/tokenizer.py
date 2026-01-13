from typing import Set
from transformers import (
    GPT2TokenizerFast,
    BartTokenizerFast,
    LlamaTokenizerFast,
    T5TokenizerFast,
    CodeGenTokenizerFast,
    PreTrainedTokenizerFast,
    GemmaTokenizerFast,
    Qwen2TokenizerFast,
    ByT5Tokenizer,
)

from transformers_cfg.tokenization.SUPPORTED_TOKENIZERS import SUPPORTED_TOKENIZERS
from transformers_cfg.tokenization.tokenizer_inspector import TokenizerInspector


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
        self.special_token_ids = set(hf_tokenizer.all_special_ids)

    def real_vocab_size(self):
        return len(self.hf_tokenizer.get_vocab())

    @classmethod
    def from_hf_tokenizer(cls, hf_tokenizer):
        """
        Create appropriate TCFG tokenizer wrapper for a HuggingFace tokenizer.

        Uses TokenizerInspector to centralize tokenizer family detection.

        Args:
            hf_tokenizer: HuggingFace tokenizer instance

        Returns:
            Appropriate TCFG_Tokenizer subclass instance

        Raises:
            AssertionError: If tokenizer type is not in SUPPORTED_TOKENIZERS
            NotImplementedError: If tokenizer family is not supported
        """
        assert (
            type(hf_tokenizer) in SUPPORTED_TOKENIZERS
        ), f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}, supported tokenizers: {SUPPORTED_TOKENIZERS}"

        # Use TokenizerInspector for centralized family detection
        inspector = TokenizerInspector(hf_tokenizer)
        family = inspector.get_family()

        if family == "gpt2" or family == "byt5":
            return TCFG_Tokenizer(hf_tokenizer)
        elif family == "llama1" or family == "llama3" or family == "t5":
            return TCFG_LlamaTokenizer(hf_tokenizer)
        elif family == "codegen":
            # Phi reuses the codegen tokenizer
            return TCFG_PhiTokenizer(hf_tokenizer)
        else:
            raise NotImplementedError(
                f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}"
            )

    # will be extended by the subclasses
    def get_special_token_ids_to_excluded(self) -> Set[int]:
        return self.special_token_ids


class TCFG_LlamaTokenizer(TCFG_Tokenizer):
    def __init__(self, hf_tokenizer):
        super().__init__(hf_tokenizer)
        self.inspector = TokenizerInspector(hf_tokenizer)

    def get_special_token_ids_to_excluded(self):
        """
        Get special token IDs to exclude from generation.

        For deepseek-coder, excludes additional special tokens beyond the standard set.
        """
        if self.inspector.is_deepseek_coder():
            # deepseek has in total 22 special tokens, with token_ids from 32000 to 32021
            # with first 13 being characters for bytes: {'õ': 32000, '÷': 32001, 'Á': 32002, 'ý': 32003, 'À': 32004, 'ÿ': 32005, 'ø': 32006, 'ú': 32007, 'þ': 32008, 'ü': 32009, 'ù': 32010, 'ö': 32011, 'û': 32012}
            # the rest are special tokens for the tokenizer: { '<｜begin▁of▁sentence｜>': 32013, '<｜end▁of▁sentence｜>': 32014, '<｜fim▁hole｜>': 32015, '<｜fim▁begin｜>': 32016, '<｜fim▁end｜>': 32017, '<pad>': 32018, '<|User|>': 32019, '<|Assistant|>': 32020, '<|EOT|>': 32021}
            added_vocab_dict = self.hf_tokenizer.get_added_vocab()
            added_tokens_id_to_excluded = set(
                [
                    token_id
                    for tok, token_id in added_vocab_dict.items()
                    if tok.startswith("<｜")
                ]
            )
            return self.special_token_ids.union(added_tokens_id_to_excluded)
        return self.special_token_ids


class TCFG_CharacterTokenizer(TCFG_Tokenizer):
    """
    Not yet used, but can be used for character level tokenization (even though rarely used in practice)
    """

    def __init__(self, hf_tokenizer):
        super().__init__(hf_tokenizer)


class TCFG_PhiTokenizer(TCFG_Tokenizer):
    def __init__(self, hf_tokenizer):
        super().__init__(hf_tokenizer)

    def real_vocab_size(self):
        return 50257  # 50 k tokens + 256 for bytes + 1 for EOS
