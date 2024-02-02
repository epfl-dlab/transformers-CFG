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
