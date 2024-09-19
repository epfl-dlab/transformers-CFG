from transformers_cfg.tokenization.SUPPORTED_TOKENIZERS import SUPPORTED_TOKENIZERS
from .ByteProxyMapping import ByteProxyMapping, LLAMAByteProxyMapper
import logging
from transformers import (
    GPT2TokenizerFast,
    BartTokenizerFast,
    T5TokenizerFast,
    CodeGenTokenizerFast,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
    GemmaTokenizerFast,
    Qwen2TokenizerFast
)

from transformers_cfg.tokenization.utils import get_tokenizer_charset

log = logging.getLogger(__name__)


class TokenizerMiddleMapping:
    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.tokenizer = tokenizer
        self.special = tokenizer.all_special_ids
        self._length = len(self.tokenizer.get_vocab())

    def __len__(self):
        return self._length

    def map(self, token_id: int, verbose=False) -> bytes:
        raise NotImplementedError("This method should be implemented in the subclass")

    @classmethod
    def from_hf_tokenizer(cls, hf_tokenizer):
        assert (
            type(hf_tokenizer) in SUPPORTED_TOKENIZERS
        ), f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}, supported tokenizers: {SUPPORTED_TOKENIZERS}"
        if isinstance(
            hf_tokenizer, (GPT2TokenizerFast, BartTokenizerFast, CodeGenTokenizerFast, Qwen2TokenizerFast)
        ):
            return GPT2TokenizerMiddleMapping(hf_tokenizer)
        elif isinstance(hf_tokenizer, (LlamaTokenizerFast, GemmaTokenizerFast)):
            # deepseek, though inheriting from LlamaTokenizerFast, is actually a GPT2TokenizerFast
            # check https://github.com/epfl-dlab/transformers-CFG/issues/72
            if hf_tokenizer.name_or_path.startswith("deepseek-ai/deepseek-coder"):
                return GPT2TokenizerMiddleMapping(hf_tokenizer)
            return LLAMA1TokenizerMiddleMapping(hf_tokenizer)
        elif isinstance(hf_tokenizer, T5TokenizerFast):
            return T5TokenizerMiddleMapping(hf_tokenizer)
        elif isinstance(
            hf_tokenizer, PreTrainedTokenizerFast
        ) and 'Meta-Llama-3' in hf_tokenizer.name_or_path:
            return GPT2TokenizerMiddleMapping(hf_tokenizer)

    @staticmethod
    def auto_infer(hf_tokenizer):
        "beta version, not sure if it will work for all cases"
        charset = get_tokenizer_charset(hf_tokenizer)
        size = len(charset)
        if size >= 256 and size < 256 + 30:
            return GPT2TokenizerMiddleMapping(hf_tokenizer)
        elif "▁" in charset:
            return LLAMA1TokenizerMiddleMapping(hf_tokenizer)
        else:
            raise NotImplementedError(
                f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}"
            )


class GPT2TokenizerMiddleMapping(TokenizerMiddleMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.byte_proxy_mapper = ByteProxyMapping(tokenizer)

    def map2proxy_token(self, token_id: int) -> str:
        # This is the case for BOS,
        if token_id in self.special:
            return ""
        # if token_id is tensor, convert it to int
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        proxy_token = self.tokenizer.convert_ids_to_tokens(token_id)
        return proxy_token

    def map(self, token_id: int, verbose=False) -> bytes:
        proxy_token = self.map2proxy_token(token_id)
        if verbose:
            log.debug(f"token_id: {token_id}, token: {proxy_token}")

        return self.byte_proxy_mapper.map(proxy_token)


class LLAMA1TokenizerMiddleMapping(TokenizerMiddleMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.last_token_id = None
        self.byte_proxy_mapper = LLAMAByteProxyMapper()

    def map(self, token_id: int, verbose=False) -> bytes:
        # we need to check if the token is at the beginning of the sentence to remove the space
        # specific to BPE
        at_bos = False
        if self.last_token_id is not None and self.last_token_id == self.bos_token_id:
            at_bos = True
        self.last_token_id = token_id

        # This is the case for BOS,
        if token_id in self.special:
            return b""
        # if token_id is tensor, convert it to int
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        proxy_token = self.tokenizer.convert_ids_to_tokens(token_id)

        token_bytes = self.byte_proxy_mapper.map(proxy_token)

        # check if the first byte is a space
        if token_bytes[0] == 32 and at_bos:
            # remove space at the beginning of the sentence
            token_bytes = token_bytes[1:]

        return token_bytes


class T5TokenizerMiddleMapping(TokenizerMiddleMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.at_bos = True
        self.byte_proxy_mapper = LLAMAByteProxyMapper()

    def map(self, token_id: int, verbose=False) -> bytes:
        # we need to check if the token is at the beginning of the sentence to remove the space
        # specific to BPE

        # This is the case for BOS,
        if token_id in self.special:
            self.at_bos = False
            return b""
        # if token_id is tensor, convert it to int
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        proxy_token = self.tokenizer.convert_ids_to_tokens(token_id)

        token_bytes = self.byte_proxy_mapper.map(proxy_token)

        # check if the first byte is a space
        if token_bytes[0] == 32 and self.at_bos:
            # remove space at the beginning of the sentence
            token_bytes = token_bytes[1:]

        self.at_bos = False
        return token_bytes
