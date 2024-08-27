#!/usr/bin/env python3

import argparse
from transformers_cfg.tokenization.utils import is_tokenizer_supported
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
import torch


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Transformers-CFG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands help")

    # Sub-command: check
    check_parser = subparsers.add_parser("check", help="Check if a model is supported")
    check_parser.add_argument(
        "model", type=str, help="The unique model name on HF hub."
    )

    # Sub-command: generate
    generate_parser = subparsers.add_parser(
        "generate", help="Generate text with grammar constraints"
    )
    generate_parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        required=True,
        help="Model identifier for loading the tokenizer and model",
    )
    generate_parser.add_argument(
        "-g",
        "--grammar_file_path",
        type=str,
        required=True,
        help="Path to the grammar file",
    )
    generate_parser.add_argument(
        "-p",
        "--prefix_prompt",
        type=str,
        required=True,
        help="Prefix prompt for generation",
    )
    generate_parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on",
    )
    generate_parser.add_argument(
        "-n",
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens to generate",
    )
    generate_parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Penalty for token repetition",
    )
    generate_parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Load the model in 4-bit mode using bitsandbytes",
    )
    generate_parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Load the model in 8-bit mode using bitsandbytes",
    )

    return parser.parse_args(args)


def check_model_support(model_name):
    # Check if the model tokenizer is supported
    if is_tokenizer_supported(model_name):
        print(f"{model_name} is supported")
        return True
    else:
        print(f"{model_name} is not supported")
        return False


def generate_text(args):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model with bitsandbytes if 8bit or 4bit flag is set
    if args.use_8bit or args.use_4bit:
        try:
            pass
        except ImportError:
            raise ImportError(
                "You need to install bitsandbytes to use 8-bit or 4-bit modes. Install it with `pip install bitsandbytes`."
            )

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=args.use_8bit,
            load_in_4bit=args.use_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=bnb_config, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id).to(args.device)

    # set special tokens in generation config
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer(
        args.prefix_prompt, add_special_tokens=False, return_tensors="pt", padding=True
    )
    input_ids = inputs["input_ids"].to(args.device)
    attention_mask = inputs["attention_mask"].to(args.device)

    # Load grammar
    with open(args.grammar_file_path, "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        logits_processor=[grammar_processor],
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=1,
    )

    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generations)


def main(args=None):
    args = parse_arguments(args)

    if args.command == "check":
        check_model_support(args.model)
    elif args.command == "generate":
        generate_text(args)


if __name__ == "__main__":
    main()


# TODO, add contrast mode where we generate text without grammar constraints and compare the outputs
# TODO, add option to save the generated text to a file
# TODO, add support to unicode grammar constraints

# TODO, add support for device selection for parsing
