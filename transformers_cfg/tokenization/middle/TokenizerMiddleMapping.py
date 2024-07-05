from .ByteProxyMapping import ByteProxyMapping, LLAMAByteProxyMapper
from transformers import AutoTokenizer
import logging

log = logging.getLogger(__name__)


def getTokenizerMiddleMapping(tokenizer):

    if "gpt2" in tokenizer.__class__.__name__.lower():
        return GPT2TokenizerMiddleMapping(tokenizer)
    elif "llama" in tokenizer.__class__.__name__.lower():
        return LLAMA1TokenizerMiddleMapping(tokenizer)
    elif "mistral" in tokenizer.__class__.__name__.lower():
        return LLAMA1TokenizerMiddleMapping(tokenizer)
    elif "t5" in tokenizer.__class__.__name__.lower():
        return T5TokenizerMiddleMapping(tokenizer)
    else:
        raise NotImplementedError(f"Unicode mapping for {tokenizer.__class__.__name__}")


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
