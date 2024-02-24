from typing import Dict, List

from transformers_cfg.utils import get_tokenizer_model_type, ints2bytes
from transformers import AutoTokenizer
import logging

log = logging.getLogger(__name__)


def get_mapping(tokenizer, unicode=False):
    log.debug(f"tokenizer type: {tokenizer.__class__.__name__}")
    log.debug(f"tokenizer model type: {get_tokenizer_model_type(tokenizer)}")
    if not unicode:
        if (
            "gpt2" in tokenizer.__class__.__name__.lower()
            or "bloom" in tokenizer.__class__.__name__.lower()
            or "pretrainedtokenizer" in tokenizer.__class__.__name__.lower()
            or "codegen" in tokenizer.__class__.__name__.lower()
            or "gptneox" in tokenizer.__class__.__name__.lower()
        ):
            return BBPEMapping(tokenizer)
        elif "t5" in tokenizer.__class__.__name__.lower():
            return BPEMapping(tokenizer)
        elif "llama" in tokenizer.__class__.__name__.lower():
            return LlamaBPEMapping(tokenizer)
        elif "xglm" in tokenizer.__class__.__name__.lower():
            return UniGramMapping(tokenizer)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer.__class__.__name__}")
    else:
        if "gpt2" in tokenizer.__class__.__name__.lower():
            return UnicodeBBPEMapping(tokenizer)
        else:
            raise NotImplementedError(
                f"Unicode mapping for {tokenizer.__class__.__name__}"
            )


class Mapping:
    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.tokenizer = tokenizer
        self.special = tokenizer.all_special_ids

    def __len__(self):
        return len(self.tokenizer.get_vocab())

    def _map(self, token_id: int) -> str:
        # This is the case for BOS,
        if token_id in self.special:
            return ""
        # if token_id is tensor, convert it to int
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        raw_token = self.tokenizer.convert_ids_to_tokens(token_id)
        return raw_token

    def map(self, token_id: int, verbose=False) -> bytes:
        token = self._map(token_id)
        if verbose:
            log.debug(f"token_id: {token_id}, token: {token}")
        return bytes(token, "utf-8")


class BBPEMapping(Mapping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _map(self, token_id: int) -> str:
        raw_token = super()._map(token_id)
        if raw_token.startswith("Ġ"):
            raw_token = raw_token.replace("Ġ", " ")
        return raw_token


class UnicodeBBPEMapping(Mapping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intermediate_encoding = UnicodeBBPEMapping.get_intermediate_encoding(
            self.tokenizer
        )

    def _map(self, token_id: int, verbose=False) -> str:
        raw_token = super()._map(token_id)
        # if raw_token.startswith("Ġ"):
        #     raw_token = raw_token.replace("Ġ", " ")
        return raw_token

    def map(self, token_id: int, verbose=False) -> bytes:
        raw_token = self._map(token_id, verbose)
        if verbose:
            log.debug(f"token_id: {token_id}, raw_token: {raw_token}")
        return self.intermediate_encoding.token2bytes(raw_token)

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

    def _map(self, token_id: int) -> str:
        raw_token = super()._map(token_id)

        # we need to check if the token is at the beginning of the sentence to remove the space
        # specific to BPE
        at_bos = False
        if self.last_token_id is not None and self.last_token_id == self.bos_token_id:
            at_bos = True
        self.last_token_id = token_id
        if raw_token.startswith("▁"):
            raw_token = raw_token.replace("▁", " ")
            if at_bos:
                # remove space at the beginning of the sentence
                raw_token = raw_token[1:]
        return raw_token


class LlamaBPEMapping(BPEMapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

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
        self.byte2cdp: Dict[int, int] = {v: k for k, v in self.cdp2byte.items()}

    def map(self, byte: int) -> int:
        assert 0 <= byte < 256, f"byte: {byte} is not in the range [0, 256)"
        return ord(self.byte2char[byte])

    def token_ids2bytes(self, token_ids: List[int]) -> bytes:
        tokens: List[str] = self.tokenizer.convert_ids_to_tokens(token_ids)
        # for token id = BOS, the token should be empty string instead of <s>
        # TODO, this may cause issues because this means that special tokens like BOS can appear at any position
        tokens = [
            "" if token in self.tokenizer.all_special_ids else token for token in tokens
        ]
        bytes: List[List[int]] = [self.token2bytes(token) for token in tokens]
        # join the bytes
        return ints2bytes(sum(bytes, []))

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
