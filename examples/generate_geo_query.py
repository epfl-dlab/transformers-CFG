import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.parser import parse_ebnf


def parse_args():
    parser = argparse.ArgumentParser(description="Generate geo query strings")
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Model ID (default: Llama 3.2 1B)",
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
    # Load model to defined device
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Handle pad_token_id (eos_token_id might be a list for some models)
    eos_token_id = model.generation_config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    model.generation_config.pad_token_id = eos_token_id

    # Load grammar
    with open(f"examples/grammars/geo_query.ebnf", "r") as file:
        grammar_str = file.read()

    parsed_grammar = parse_ebnf(grammar_str)
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

    # Note: We cannot use valid_token_start_idx here because we need to generate
    # both unconstrained and constrained outputs with the same input_ids.
    # For proper grammar constraint, the prompt should NOT be included.
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prompts = [
        "which state contains most rivers ? ",
        "number of citizens in boulder ? ",
        "what are the major cities of the us ? ",
        "what is the smallest city in washington ? ",
        "how many states border colorado and border new mexico ? ",
    ]

    input_ids = tokenizer(
        prompts, add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"].to(
        device
    )  # Move input_ids to the same device as model

    n_examples = input_ids.shape[0]
    prompt_lengths = input_ids.shape[1]  # Track prompt length in tokens

    max_new_tokens = 50
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

    parsed_grammar = parse_ebnf(grammar_str)
    string_grammar = StringRecognizer(
        parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"]
    )

    # Decode full outputs for display
    full_generations = tokenizer.batch_decode(
        unconstrained_output, skip_special_tokens=True
    ) + tokenizer.batch_decode(constrained_output, skip_special_tokens=True)

    # Decode only the generated parts (excluding prompt) for validation
    unconstrained_only = tokenizer.batch_decode(
        unconstrained_output[:, prompt_lengths:], skip_special_tokens=True
    )
    constrained_only = tokenizer.batch_decode(
        constrained_output[:, prompt_lengths:], skip_special_tokens=True
    )

    print()
    for i in range(n_examples):
        print(f"Unconstrained: {full_generations[i]}")
        print(f"Constrained: {full_generations[i + n_examples]}")

        # Validate using the generated-only part (no prompt)
        generated_text = constrained_only[i]
        print(f"Generated part only: {generated_text}")
        print(
            f"The constrained generation matches the grammar: {string_grammar._accept_string(generated_text)}"
        )
        print(
            f"The generated prefix matches the grammar: {string_grammar._accept_prefix(generated_text)}"
        )
        print()


if __name__ == "__main__":
    main()


##########################
# Example output:
#
# Unconstrained: how many states border colorado and border new mexico ? 1.
# - How long is the drive from denver to albuquerque? The distance between Denver, Colorado (CO) & Alburqueque New Mexico(NM). Driving directions for your road trip or vacation: Get driving
# Constrained: how many states border colorado and border new mexico ? answer(smallest_one(area_1(stateid('colorado'))))
#
##########################
