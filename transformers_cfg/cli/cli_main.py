#!/usr/bin/env python3

import argparse
from importlib import import_module
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
        "--prompt",
        type=str,
        required=True,
        help="Prompt for generation",
    )
    generate_parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
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
    generate_parser.add_argument(
        "--no_contrast_mode",
        action="store_true",
        help="Disable contrast mode (enabled by default)",
    )
    generate_parser.add_argument(
        "--save_to",
        type=str,
        help="File path to save the generated text",
    )
    generate_parser.add_argument(
        "--use_mlx",
        action="store_true",
        help="Use MLX on max to speed up generation",
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

    # Load grammar
    with open(args.grammar_file_path, "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    if args.use_mlx:
        try:
            import_module("mlx_lm")
        except ImportError:
            raise ImportError(
                "You need to install mlx to use MLX. Install it with `pip install 'git+https://github.com/nathanrchn/mlx-examples.git@logits_processor#subdirectory=llms'`."
            )
        
        import numpy as np
        import mlx.core as mx
        from mlx_lm import load, stream_generate

        model, _ = load(args.model_id)

        def logits_processor(input_ids: mx.array, logits: mx.array) -> mx.array:
            torch_input_ids = torch.tensor(np.array(input_ids[None, :]), device=args.device)
            torch_logits = torch.tensor(np.array(logits), device=args.device)

            torch_processed_logits = grammar_processor(torch_input_ids, torch_logits)
            return mx.array(torch_processed_logits.cpu().numpy())

        generation_stream = stream_generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            logits_processor=logits_processor
        )

        # print prompt first in color
        print("\033[92m" + "Prompt:" + args.prompt + "\033[0m")

        print("\033[94m" + "Constrained Generation:" + "\033[0m")
        for token in generation_stream:
            print(token, end="", flush=True)

        print()
        return

    # Load the model with bitsandbytes if 8bit or 4bit flag is set
    if args.use_8bit or args.use_4bit:
        try:
            import_module("bitsandbytes")
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
        args.prompt, add_special_tokens=False, return_tensors="pt", padding=True
    )
    input_ids = inputs["input_ids"].to(args.device)
    attention_mask = inputs["attention_mask"].to(args.device)

    # Generate with grammar constraints
    constrained_output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        logits_processor=[grammar_processor],
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=1,
    )

    # remove prefix from the output
    constrained_output = constrained_output[:, len(input_ids[0]) :]

    constrained_generations = tokenizer.batch_decode(
        constrained_output, skip_special_tokens=True
    )

    # print prompt first in color
    print("\033[92m" + "Prompt:" + args.prompt + "\033[0m")

    # Store results for optional file output
    result = f"Prompt: {args.prompt}\n\n"

    # Generate without grammar constraints (if contrast mode is enabled)
    if not args.no_contrast_mode:
        unconstrained_output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=1,
        )
        # remove prefix from the output
        unconstrained_output = unconstrained_output[:, len(input_ids[0]) :]
        unconstrained_generations = tokenizer.batch_decode(
            unconstrained_output, skip_special_tokens=True
        )

        # Print results in different colors
        print("\033[91m" + "Unconstrained Generation:" + "\033[0m")
        result += "Unconstrained Generation:\n"
        for generation in unconstrained_generations:
            print(generation)
            result += generation + "\n"

    print("\033[94m" + "Constrained Generation:" + "\033[0m")
    result += "Constrained Generation:\n"
    for generation in constrained_generations:
        print(generation)
        result += generation + "\n"

    # Save to file if save_to is provided
    if args.save_to:
        with open(args.save_to, "w") as f:
            f.write(result)
        print(f"\nResults saved to {args.save_to}")


def main(args=None):
    args = parse_arguments(args)

    if args.command == "check":
        check_model_support(args.model)
    elif args.command == "generate":
        generate_text(args)


if __name__ == "__main__":
    main()

# TODO, add support for device selection for parsing
