import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.parser import parse_ebnf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate chain of thought arithmentic strings"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        help="Model ID",
    )
    parser.add_argument("--device", type=str, help="Device to put the model on")
    return parser.parse_args()


def main():
    args = parse_args()
    model_id = args.model_id

    # Detect if GPU is available, otherwise use CPU
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Load model to defined device
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Load grammar
    with open(f"examples/grammars/chain_of_thought_arithmetic.ebnf", "r") as file:
        grammar_str = file.read()

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prompts = [
        "179*12+34=",  # no CoT
        "think step-by-step, 12+7*19=12+133=145 >>> 145; 7*8+6*9=56+54=110 >>> 110; 179*12+34=",  # CoT
    ]

    input_ids = tokenizer(
        prompts, add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"].to(
        device
    )  # Move input_ids to the same device as model

    n_examples = input_ids.shape[0]

    max_new_tokens = 30

    unconstrained_output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.9,
        num_return_sequences=1,
    )

    constrained_output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        logits_processor=[grammar_processor],
        repetition_penalty=1.9,
        num_return_sequences=1,
    )

    # decode outputs (possibly of different lengths across decoding modes)
    generations = tokenizer.batch_decode(
        unconstrained_output, skip_special_tokens=True
    ) + tokenizer.batch_decode(constrained_output, skip_special_tokens=True)

    parsed_grammar = parse_ebnf(grammar_str)
    string_grammar = StringRecognizer(
        parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"]
    )

    print()
    for i in range(n_examples):
        print(f"Unconstrained: {generations[i]}")
        constrained_generation = generations[i + n_examples]
        print(f"Constrained: {constrained_generation}")
        print(
            f"The constrained generation matches the grammar: {string_grammar._accept_string(constrained_generation[len(prompts[i]):])}"
        )
        print(
            f"The generated prefix matches the grammar: {string_grammar._accept_prefix(constrained_generation[len(prompts[i]):])}"
        )
        print()


if __name__ == "__main__":
    main()

##########################
# Example output (no chain of thought):
# Unconstrained:
# 179*12+34=0,
# -568. Вторемьте в некоторых другие позиции (включая и
#
# Constrained:
# 179*12+34=0;
# The constrained generation matches the grammar: True
# The generated prefix matches the grammar: True
#
# Example output (with chain of thought):
# Unconstrained:
# think step-by-step, 12+7*19=12+133=145 >>> 145; 7*8+6*9=56+54=110 >>> 110; 179*12+34=2148.0 + 117 = <<< error: invalid type comparison >>>;
# ``` | ```vbnet
# '
# Constrained:
# think step-by-step, 12+7*19=12+133=145 >>> 145; 7*8+6*9=56+54=110 >>> 110; 179*12+34=2148+34=2182 >>> 2182;
# The constrained generation matches the grammar: True
# The generated prefix matches the grammar: True
##########################
