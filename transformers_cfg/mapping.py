from transformers_cfg.utils import get_tokenizer_model_type

import logging

log = logging.getLogger(__name__)


def get_mapping(tokenizer):
    log.debug(f"tokenizer type: {tokenizer.__class__.__name__}")
    log.debug(f"tokenizer model type: {get_tokenizer_model_type(tokenizer)}")
    if (
        "gpt2" in tokenizer.__class__.__name__.lower()
        or "bloom" in tokenizer.__class__.__name__.lower()
        or "pretrainedtokenizer" in tokenizer.__class__.__name__.lower()
        or "codegen" in tokenizer.__class__.__name__.lower()
        or "gptneox" in tokenizer.__class__.__name__.lower()
    ):
        return BBPEMapping(tokenizer)
    elif (
        "t5" in tokenizer.__class__.__name__.lower()
        or "phi" in tokenizer.__class__.__name__.lower()
    ):
        return BPEMapping(tokenizer)
    elif "llama" in tokenizer.__class__.__name__.lower():
        return LlamaBPEMapping(tokenizer)
    elif (
        "xlnet" in tokenizer.__class__.__name__.lower()
        or "xglm" in tokenizer.__class__.__name__.lower()
    ):
        return UniGramMapping(tokenizer)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer.__class__.__name__}")


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

    def map(self, token_id: int) -> bytes:
        token = self._map(token_id)
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
