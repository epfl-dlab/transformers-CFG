from abc import ABC, abstractmethod

import torch
from transformers_cfg.tokenization.SUPPORTED_TOKENIZERS import SUPPORTED_TOKENIZERS
from .ByteProxyMapping import ByteProxyMapping, LLAMAByteProxyMapping
import logging
from transformers import (
    GPT2TokenizerFast,
    BartTokenizerFast,
    T5TokenizerFast,
    CodeGenTokenizerFast,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
    GemmaTokenizerFast,
    Qwen2TokenizerFast,
    ByT5Tokenizer,
)

from transformers_cfg.tokenization.utils import get_tokenizer_charset

log = logging.getLogger(__name__)


class Token2ByteMapping(ABC):
    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.tokenizer = tokenizer
        self.special = tokenizer.all_special_ids
        self._length = len(self.tokenizer.get_vocab())

    def __len__(self):
        return self._length

    @abstractmethod
    def map(self, token_id: int, verbose=False) -> bytes:
        pass

    @classmethod
    def from_hf_tokenizer(cls, hf_tokenizer):
        assert (
            type(hf_tokenizer) in SUPPORTED_TOKENIZERS
        ), f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}, supported tokenizers: {SUPPORTED_TOKENIZERS}"
        if isinstance(
            hf_tokenizer,
            (
                GPT2TokenizerFast,
                BartTokenizerFast,
                CodeGenTokenizerFast,
                Qwen2TokenizerFast,
            ),
        ):
            return GPT2Token2ByteMapping(hf_tokenizer)
        elif isinstance(hf_tokenizer, (LlamaTokenizerFast, GemmaTokenizerFast)):
            # deepseek, though inheriting from LlamaTokenizerFast, is actually a GPT2TokenizerFast
            # check https://github.com/epfl-dlab/transformers-CFG/issues/72
            if "deepseek-coder" in hf_tokenizer.name_or_path:
                return GPT2Token2ByteMapping(hf_tokenizer)
            return LLAMA1Token2ByteMapping(hf_tokenizer)
        elif isinstance(hf_tokenizer, T5TokenizerFast):
            return T5Token2ByteMapping(hf_tokenizer)
        elif (
            isinstance(hf_tokenizer, PreTrainedTokenizerFast)
            and "Llama-3"
            in hf_tokenizer.name_or_path  # this includes llama-3/llama-3.1/llama-3.2/llama-3.3
        ):
            return GPT2Token2ByteMapping(hf_tokenizer)
        elif isinstance(hf_tokenizer, ByT5Tokenizer):
            return ByT5Token2ByteMapping(hf_tokenizer)
        else:
            raise NotImplementedError(
                f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}"
            )

    @staticmethod
    def auto_infer(hf_tokenizer):
        "beta version, not sure if it will work for all cases"
        charset = get_tokenizer_charset(hf_tokenizer)
        size = len(charset)
        if size >= 256 and size < 256 + 30:
            return GPT2Token2ByteMapping(hf_tokenizer)
        elif "▁" in charset:
            return LLAMA1Token2ByteMapping(hf_tokenizer)
        else:
            raise NotImplementedError(
                f"Tokenizer not supported: {hf_tokenizer.__class__.__name__}"
            )


class GPT2Token2ByteMapping(Token2ByteMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.byte_proxy_mapping = ByteProxyMapping(tokenizer)

    def map2proxy_token(self, token_id: int) -> str:
        # This is the case for BOS,
        if token_id in self.special:
            return ""
        # if token_id is tensor, convert it to int
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item()
        proxy_token = self.tokenizer.convert_ids_to_tokens(token_id)
        return proxy_token

    def map(self, token_id: int, verbose=False) -> bytes:
        proxy_token = self.map2proxy_token(token_id)
        if verbose:
            log.debug(f"token_id: {token_id}, token: {proxy_token}")

        return self.byte_proxy_mapping.map(proxy_token)


class LLAMA1Token2ByteMapping(Token2ByteMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.last_token_id = None
        self.byte_proxy_mapping = LLAMAByteProxyMapping()

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
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item()
        proxy_token = self.tokenizer.convert_ids_to_tokens(token_id)

        token_bytes = self.byte_proxy_mapping.map(proxy_token)

        # check if the first byte is a space
        if token_bytes[0] == 32 and at_bos:
            # remove space at the beginning of the sentence
            token_bytes = token_bytes[1:]

        return token_bytes


class T5Token2ByteMapping(Token2ByteMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.at_bos = True
        self.byte_proxy_mapper = LLAMAByteProxyMapping()

    def map(self, token_id: int, verbose=False) -> bytes:
        # we need to check if the token is at the beginning of the sentence to remove the space
        # specific to BPE

        # This is the case for BOS,
        if token_id in self.special:
            self.at_bos = False
            return b""
        # if token_id is tensor, convert it to int
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item()
        proxy_token = self.tokenizer.convert_ids_to_tokens(token_id)

        token_bytes = self.byte_proxy_mapper.map(proxy_token)

        # check if the first byte is a space
        if token_bytes[0] == 32 and self.at_bos:
            # remove space at the beginning of the sentence
            token_bytes = token_bytes[1:]

        self.at_bos = False
        return token_bytes


class ByT5Token2ByteMapping(Token2ByteMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.token_id_to_bytes = {}

    def map(self, token_id: int, verbose=False) -> bytes:
        # By inspecting the token vocab, we can see that the first 3 tokens are special tokens
        # and the tokens after 258 are also special tokens
        # only the tokens between 3 and 258 are valid tokens, 256 bytes
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item()
        if token_id in self.token_id_to_bytes:
            return self.token_id_to_bytes[token_id]
        if 3 <= token_id <= 258:
            return ord(self.tokenizer.convert_ids_to_tokens(token_id)).to_bytes(
                1, "big"
            )
        else:
            # return empty bytes for special tokens
            return bytes()
