import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.parser import parse_ebnf

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SMILES strings")
    parser.add_argument("--model_id", type=str, default="/dlabdata1/llm_hub/Mistral-7B-v0.1", help="Model ID")
    parser.add_argument("--device", type=str, help="Device to put the model on")
    return parser.parse_args()


def main():
    args = parse_args()
    model_id = args.model_id

    # Detect if GPU is available, otherwise use CPU
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # Load model to defined device
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device) 

    # Load grammar
    with open(f"examples/grammars/SMILES/geo_query.ebnf", "r") as file:
        grammar_str = file.read()

    parsed_grammar = parse_ebnf(grammar_str)
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    propmts = [
        "which state contains most rivers ? ",
        "number of citizens in boulder ? ",
        "what are the major cities of the us ? ",
        "what is the smallest city in washington ? ", 
        "how many states border colorado and border new mexico ? ",
    ]

    input_ids = tokenizer(
        propmts, add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"].to(
        device
    )  # Move input_ids to the same device as model
    
    n_examples = input_ids.shape[0]

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

    # decode output
    generations = tokenizer.batch_decode(torch.concat([unconstrained_output, constrained_output]), skip_special_tokens=True)
    for generation, gen_type in zip(generations, ['unconstrained:'] * n_examples + ['constrained:'] * n_examples):
        print("_" * 10)
        print(gen_type)
        print(generation)
        print("_" * 10)


if __name__ == "__main__":
    main()