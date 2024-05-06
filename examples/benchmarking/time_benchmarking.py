import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.parser import parse_ebnf
import time
import os
import sys
from dataclasses import dataclass


@dataclass
class BenchmarkingArguments:
    grammar_filepath: str
    prompt: str
    max_new_tokens: int = 50
    model_id: str = "/dlabdata1/llm_hub/Mistral-7B-v0.1"
    device: str = "cpu"


MAX_NEW_TOKEN_PLACEHOLDER = "<MAX_NEW_TOKENS>"


def parse_args():
    raw_args = sys.argv[1:]
    n_passed = len(raw_args)
    if n_passed < 2:
        print("Usage: python time_benchmarking.py <grammar_filepath> <prompt>")
        return
    if n_passed > 2:
        raw_args[2] = int(raw_args[2])
    args = BenchmarkingArguments(*raw_args)
    return args


def main():
    args = parse_args()

    model_id = args.model_id

    # Detect if GPU is available, otherwise use CPU
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}, max new tokens: {args.max_new_tokens}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # Load model to defined device
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Load grammar
    with open(args.grammar_filepath, "r") as file:
        grammar_str = file.read()

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    args.prompt = args.prompt.replace(
        MAX_NEW_TOKEN_PLACEHOLDER, str(args.max_new_tokens)
    )

    input_ids = tokenizer(
        [args.prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"].to(device)

    unconstrained_st = time.perf_counter()
    unconstrained_output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=1,
    )
    unconstrained_tot = time.perf_counter() - unconstrained_st

    constrained_st = time.perf_counter()
    constrained_output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        logits_processor=[grammar_processor],
        num_return_sequences=1,
    )

    constrained_tot = time.perf_counter() - constrained_st
    print(f"Unconstrained time: {unconstrained_tot:.2f}")
    print(f"Constrained time: {constrained_tot:.2f}")

    # decode outputs (possibly of different lengths across decoding modes)
    generations = tokenizer.batch_decode(
        unconstrained_output, skip_special_tokens=True
    ) + tokenizer.batch_decode(constrained_output, skip_special_tokens=True)
    print()

    n_examples = len(input_ids)
    for i in range(n_examples):
        unconstrained_generation = generations[i]
        constrained_generation = generations[i + n_examples]

        for generation, generation_type in zip(
            [unconstrained_generation, constrained_generation],
            ["unconstrained", "constrained"],
        ):
            print(f"The {generation_type} generation:\n{generation}")
            print()


if __name__ == "__main__":
    main()
