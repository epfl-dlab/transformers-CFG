import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
from transformers_cfg.generation.logits_process import (
    # LogitsProcessorList,
    IncrementalGrammarConstraint,
    GrammarConstrainedLogitsProcessor,  # default cfg
    BlockBadStateLogitsProcessor,       # our "inverse" processor
)

if __name__ == "__main__":
    hf_logging.set_verbosity_error()          # silence HF warnings

    # set up isGood or isBad from command line
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--isGood", action="store_true", help="Use the good constraint")
    group.add_argument("--isBad", action="store_true", help="Use the inverse (bad) constraint")

    args = parser.parse_args()
    isGood = args.isGood
    isBad = args.isBad

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model and tokenizer
    # model_id = "facebook/opt-125m"        # known error of mismatch b/w model.config.vocab_size and tokenizer.vocab_size
    model_id = "gpt2"       # instead, another small model with a tokenizer supported by transformers_cfg
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # "cat" and "fish" are allowed, "dog" is an error state, indicated by the preceding "-"
    grammar_str = """
    root   ::= "The animal is a " animal "."
    animal ::= "cat" | "fish" | -"dog"
    """

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    if isGood:
        processor = GrammarConstrainedLogitsProcessor(grammar) # default cfg will ignore "-" and accept dog as valid
    elif isBad:
        processor = BlockBadStateLogitsProcessor(grammar) # our "inverse" processor will block dog as valid
    else: # sanity check
        raise ValueError("Must specify either --isGood or --isBad")

    prompts = [
        'The text says, "The animal is a dog." The answer is obvious. ',
        'I\'m going to say "The animal is a fish." Here I go! '
    ]
    input_ids = tokenizer(
        prompts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    ).to(device)["input_ids"]

    # generate with the grammar constraint
    outputs = model.generate(
        input_ids,
        max_length=30,
        logits_processor=[processor],
        num_return_sequences=1,
    )

    generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("#" * 80)
    print("Generated text:")
    for txt in generations:
        print(txt)
