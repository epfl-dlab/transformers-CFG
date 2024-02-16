import json
import warnings
from typing import List

from termcolor import colored


def ints2bytes(sequence: List[int]) -> bytes:
    # check in the range of 0-255
    for item in sequence:
        if not 0 <= item <= 255:
            raise ValueError(f"item: {item} is not in the range [0, 255]")
    return bytes(sequence)


def bytes2ints(byte_sequence: bytes) -> List[int]:
    return list(byte_sequence)


def intervals_intersect(low1, high1, low2, high2):
    """
    Check if two intervals [low1, high1] and [low2, high2] intersect.

    :param high1: High bound of the first interval.
    :param low1: Low bound of the first interval.
    :param high2: High bound of the second interval.
    :param low2: Low bound of the second interval.
    :return: True if the intervals intersect, False otherwise.
    """
    # Check if one interval is completely to the right of the other
    if low1 > high2 or low2 > high1:
        return False

    # If the above condition is not met, the intervals intersect
    return True


def pprint_token_ids(tokenizer, token_ids=None, text=None):
    if token_ids is None and text is None:
        raise ValueError("Either token_ids or text should be provided")
    if token_ids is None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    special_token_ids = tokenizer.all_special_ids
    special_tokens = tokenizer.all_special_tokens
    special_id2token = {
        id: token for id, token in zip(special_token_ids, special_tokens)
    }
    # loop over token_ids and color the special tokens
    colored_token_ids = []

    for token_id in token_ids:
        if token_id in special_id2token:
            colored_token_ids.append(colored(token_id, "red", attrs=["bold"]))
        else:
            colored_token_ids.append(str(token_id))
    colored_token_ids_str = [str(item) for item in colored_token_ids]
    print("[" + ", ".join(colored_token_ids_str) + "]")


def get_tokenizer_model_type(model: str = "gpt2"):
    """
    reference https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_fast.py#L729
    :param model:
    :return: BPE, Unigram, WordPiece, WordLevel
    SentencePiece is used in conjunction with Unigram
    """
    from transformers import AutoTokenizer

    # if the tokenizer is not in the repo, it will raise OSError
    # OSError: Can't load tokenizer for 'xxx'
    # This happens when the model reuses the tokenizer of another model
    if type(model) == str:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
            # check if the tokenizer is fast
        except OSError:
            return None
    else:
        tokenizer = model

    if not tokenizer.is_fast:
        raise ValueError(f"The tokenizer {model} is not fast tokenizer")
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    model_type = tokenizer_json["model"]["type"]
    if (
        model_type == "BPE"
        and tokenizer_json["pre_tokenizer"] is not None
        and (
            tokenizer_json["pre_tokenizer"]["type"] == "ByteLevel"
            or tokenizer_json["pre_tokenizer"]["pretokenizers"][1]["type"]
            == "ByteLevel"
        )
    ):
        model_type = "ByteLevelBPE"
    return model_type
