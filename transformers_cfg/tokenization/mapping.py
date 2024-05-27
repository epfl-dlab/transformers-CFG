from typing import Dict, List

from transformers_cfg.utils import get_tokenizer_model_type, ints2bytes
from transformers import AutoTokenizer
import re
import logging

log = logging.getLogger(__name__)


def get_mapping(tokenizer, unicode=False):
    log.debug(f"tokenizer type: {tokenizer.__class__.__name__}")
    log.debug(f"tokenizer model type: {get_tokenizer_model_type(tokenizer)}")
    tokenizer_name = tokenizer.__class__.__name__.lower()
    if not unicode:
        if re.match(
            r"gpt2|bloom|pretrainedtokenizer|codegen|gptneox|Llama-3", tokenizer_name
        ):
            return BBPEMapping(tokenizer)
        elif re.match(r"t5|Phi-3", tokenizer_name):
            return BPEMapping(tokenizer)
        elif "llama" in tokenizer_name:
            return LlamaBPEMapping(tokenizer)
        elif "xglm" in tokenizer_name:
            return UniGramMapping(tokenizer)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer.__class__.__name__}")
    else:
        if "gpt2" in tokenizer_name:
            return UnicodeBBPEMapping(tokenizer)
        else:
            raise NotImplementedError(
                f"Unicode mapping for {tokenizer.__class__.__name__}"
            )


class ReplacePrefixMixin:
    def __init__(self, prefix):
        self.prefix = prefix

    def _replace_prefix(self, token: str) -> str:
        if token.startswith(self.prefix):
            return token.replace(self.prefix, "", 1)
        return token


class Mapping:
    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.tokenizer = tokenizer
        self.special = tokenizer.all_special_ids
        self._length = len(self.tokenizer.get_vocab())

    def __len__(self):
        return self._length

    def _map(self, token_id: int) -> str:
        # if token_id is tensor, convert it to int
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        # This is the case for BOS,
        if token_id in self.special:
            return ""
        raw_token = self.tokenizer.convert_ids_to_tokens(token_id)
        return raw_token

    def _encode(self, token: str) -> bytes:
        return bytes(token, "utf-8")

    def map(self, token_id: int, verbose=False) -> bytes:
        token = self._map(token_id)
        if verbose:
            log.debug(f"token_id: {token_id}, token: {token}")
        return self._encode(token)


class BBPEMapping(Mapping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _map(self, token_id: int) -> str:
        raw_token = super()._map(token_id)
        if raw_token.startswith("Ġ"):
            raw_token = raw_token.replace("Ġ", " ", 1)
        return raw_token


class UnicodeBBPEMapping(Mapping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intermediate_encoding = UnicodeBBPEMapping.get_intermediate_encoding(
            self.tokenizer
        )

    def _encode(self, token: str) -> bytes:
        return self.intermediate_encoding.token2bytes(token)

    @staticmethod
    def get_intermediate_encoding(tokenizer):
        if "gpt2" in tokenizer.__class__.__name__.lower():
            return ByteEncoding(tokenizer)
        else:
            return None


class BPEMapping(Mapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.last_token_id = None

    def _check_bos_token(self, token_id: int) -> bool:
        # specific to BPE
        at_bos = self.last_token_id is None
        self.last_token_id = token_id if token_id != self.eos_token_id else None
        return at_bos

    def _map(self, token_id: int) -> str:
        raw_token = super()._map(token_id)
        # we need to check if the token is at the beginning of the sentence to remove the space
        # specific to BPE
        at_bos = self._check_bos_token(token_id)
        if raw_token.startswith("▁"):
            raw_token = raw_token.replace("▁", " ", 1)
            if at_bos:
                # remove space at the beginning of the sentence
                raw_token = raw_token[1:]
        return raw_token


class LlamaBPEMapping(BPEMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def _check_bos_token(self, token_id: int) -> bool:
        at_bos = self.last_token_id and (self.last_token_id == self.bos_token_id)
        self.last_token_id = token_id
        return at_bos

    def _map(self, token_id: int) -> str:
        raw_token = super()._map(token_id)
        # if the token is hex, token is a string like "<0x00>"
        # first 256 tokens are hex
        if raw_token.startswith("<0x"):
            hex_value = raw_token[4:-1]
            raw_token = chr(int(hex_value, 16))
        return raw_token


class WordPieceMapping(Mapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def map(self, token_id: int) -> bytes:
        if token_id in self.special:
            return bytes()
        return bytes(
            self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
            "utf-8",
        )


class UniGramMapping(Mapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def map(self, token_id: int) -> bytes:
        if token_id in self.special:
            return bytes()
        return bytes(
            self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
            "utf-8",
        )


class XGLMUniGramMapping(Mapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token_id = tokenizer.eos_token_id
        self.eos_token_id = None


class ByteEncoding:
    def __init__(self, tokenizer):
        # check if the tokenizer is fast, if so, convert it to slow
        if tokenizer.is_fast:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer.name_or_path, use_fast=False
            )
        self.tokenizer = tokenizer
        self.byte2char: Dict[int, str] = tokenizer.byte_encoder
        self.char2byte: Dict[str, int] = tokenizer.byte_decoder
        # code point to byte
        self.cdp2byte: Dict[int, int] = {ord(c): b for c, b in self.char2byte.items()}
        self.byte2cdp: Dict[int, int] = {b: c for c, b in self.cdp2byte.items()}

    def map(self, byte: int) -> int:
        assert 0 <= byte < 256, f"byte: {byte} is not in the range [0, 256)"
        return self.byte2cdp[byte]

    def token_ids2bytes(self, token_ids: List[int]) -> bytes:
        tokens: List[str] = self.tokenizer.convert_ids_to_tokens(token_ids)
        # for token id = BOS, the token should be empty string instead of <s>
        # TODO, this may cause issues because this means that special tokens like BOS can appear at any position
        tokens = [
            "" if token in self.tokenizer.all_special_ids else token for token in tokens
        ]
        bytes_per_token: List[List[int]] = [self.token2bytes(token) for token in tokens]
        # join the bytes
        bytes = sum(bytes_per_token, [])
        # verify range and convert to bytes
        bytes = ints2bytes(bytes)
        return bytes

    # Not used
    def token_id2bytes(self, token_id: int) -> bytes:
        token: str = self.tokenizer.convert_ids_to_tokens(token_id)
        return self.token2bytes(token)

    def token2bytes(self, token: str) -> bytes:
        # import pdb; pdb.set_trace()
        bytes_seq: List[int] = [self.char2byte[c] for c in token]
        return bytes(bytes_seq)


if __name__ == "__main__":
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    gpt2_tokenizer.encode("榴莲")
    # [162, 99, 112, 164, 236, 110]

    mapping = get_mapping(gpt2_tokenizer)

    x = mapping.map(162)
    # b'\xef\xbf\xbd'
    x = mapping.map(99)

    ##################################

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)

    token_ids = gpt2_tokenizer.encode("榴莲")

    print(token_ids)
    utf8_encoding = "榴莲".encode("utf-8")
    print("utf8_encoding: ", utf8_encoding)
    mapping = ByteEncoding(gpt2_tokenizer)
    print(mapping.token_ids2bytes(token_ids))
    # b'\xef\xbf\xbd'

    ###################################

    gpt2_tokenizer.encode("´")
