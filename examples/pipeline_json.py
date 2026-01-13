import torch
import argparse
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate json strings with huggingface pipelining"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
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
    with open(f"examples/grammars/json.ebnf", "r") as file:
        grammar_str = file.read()

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Initialize pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_length=50,
        batch_size=2,
    )
    # # outputs = pipe("This is a valid json string for http request:", do_sample=False, max_length=50)
    generations = pipe(
        [
            "This is a valid json string for http request: ",
            "This is a valid json string for shopping cart: ",
        ],
        do_sample=False,
        logits_processor=[grammar_processor],
    )

    print(generations)

    """
    This is a valid json string for http request: {"name":"John","age":30,"city":"New York"}
    This is a valid json string for shopping cart: {"items":[{"id":"1","quantity":"1"},{"id":"2","quantity":"2"}]}
    """


if __name__ == "__main__":
    main()
