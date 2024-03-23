import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.parser import parse_ebnf


def parse_args():
    parser = argparse.ArgumentParser(description="Generate calflow strings")
    parser.add_argument(
        "--model-id",
        type=str,
        default="/dlabdata1/llm_hub/Mistral-7B-v0.1",
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
    # Load model to defined device
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Load grammar
    with open(f"examples/grammars/calflow.ebnf", "r") as file:
        grammar_str = file.read()

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prompts = [
        'Generate 3 CalFlow strings: 1.(Yield (toRecipient (CurrentUser))) 2.(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (Event.subject_? (?= "choose the meeting"))))) 3.'
    ]

    input_ids = tokenizer(
        prompts, add_special_tokens=False, return_tensors="pt", padding=True
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
# Example output:
#
# Unconstrained: Generate 3 CalFlow strings: 1.(Yield (toRecipient (CurrentUser))) 2.(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (Event.subject_? (?= "choose the meeting"))))) 3.((yielder) ((reciever)) (((event-type)? ("create")(("prefight" ?))))
# ```
# Constrained: Generate 3 CalFlow strings: 1.(Yield (toRecipient (CurrentUser))) 2.(Yield (CreateCommitEventWrapper (CreatePreflightEventWrapper (Event.subject_? (?= "choose the meeting"))))) 3.(Yield (Path.apply "create"))
# The constrained generation matches the grammar: True
# The generated prefix matches the grammar: True
##########################
