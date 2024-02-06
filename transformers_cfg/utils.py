import json
import warnings

from termcolor import colored


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
    print("[" + ", ".join(colored_token_ids) + "]")


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
