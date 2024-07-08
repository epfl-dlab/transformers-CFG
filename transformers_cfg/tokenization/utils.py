import inspect
from typing import Dict
from transformers_cfg.tokenization.SUPPORTED_TOKENIZERS import SUPPORTED_TOKENIZERS


def replace_hex(match):
    hex_value = match.group(1)
    return chr(int(hex_value, 16))


# This will collect all imported classes from the current module (globals())
def get_imported_tokenizer_classes(module_globals) -> Dict[str, type]:
    return {
        name: obj
        for name, obj in module_globals.items()
        if inspect.isclass(obj) and name.endswith("TokenizerFast")
    }


def get_tokenizer_real_class(hf_tokenizer):
    return hf_tokenizer.__class__


def is_tokenizer_supported(hf_tokenizer_or_name):
    if isinstance(hf_tokenizer_or_name, str):
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_or_name)
    else:
        hf_tokenizer = hf_tokenizer_or_name
    return get_tokenizer_real_class(hf_tokenizer) in SUPPORTED_TOKENIZERS
