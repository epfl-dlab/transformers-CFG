import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
import logging

logging.basicConfig(level=logging.DEBUG)


def main(args):

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id)

    # Load grammar
    with open(args.grammar_file_path, "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prefix = args.prefix_prompt
    input_ids = tokenizer(
        prefix, add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]

    output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=20,
        logits_processor=[grammar_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
    )
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)

    print(generations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with grammar constraints."
    )
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        required=True,
        help="Model identifier for loading the tokenizer and model",
        default="gpt2",
    )
    parser.add_argument(
        "-g",
        "--grammar_file_path",
        type=str,
        required=True,
        help="Path to the grammar file (supports both relative and absolute paths)",
    )
    parser.add_argument(
        "-p",
        "--prefix_prompt",
        type=str,
        required=True,
        help="Prefix prompt for generation",
    )

    args = parser.parse_args()
    main(args)
