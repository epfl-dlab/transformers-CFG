from typing import Dict, List

from transformers_cfg.tokenization.middle.TokenizerMiddleMapping import (
    GPT2TokenizerMiddleMapping,
    LLAMA1TokenizerMiddleMapping,
)
from transformers_cfg.utils import get_tokenizer_model_type, ints2bytes
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
        return LLAMA1TokenizerMiddleMapping(tokenizer)
    else:
        raise NotImplementedError(f"Unicode mapping for {tokenizer.__class__.__name__}")
