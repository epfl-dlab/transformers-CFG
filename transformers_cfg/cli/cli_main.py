#!/usr/bin/env python3

import argparse
from transformers_cfg.tokenization.utils import is_tokenizer_supported


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        description="Get the tokenizer class for a given model name."
    )
    parser.add_argument(
        "--model", "-m", type=str, help="The unique model name on HF hub."
    )
    return parser.parse_args(args)


def check_model_support(model_name):
    # Check if the model tokenizer is supported

    # for now the only condition is that the tokenizer is in SUPPORTED_TOKENIZERS
    # maybe there will be more conditions in the future
    if is_tokenizer_supported(model_name):
        print(f"{model_name} is supported")
        return True
    else:
        print(f"{model_name} is not supported")
        return False


def main(args=None):
    args = parse_arguments(args)
    result = check_model_support(args.model, args.tokenizer_type, args.config_path)
    print(result)


if __name__ == "__main__":
    main()
