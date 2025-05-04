import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
from transformers.generation.logits_process import LogitsProcessorList
from transformers_cfg.generation.logits_process import (
    IncrementalGrammarConstraint,
    BlockBadStateLogitsProcessor,
)

if __name__ == "__main__":
    hf_logging.set_verbosity_error()          # silence HF warnings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # --- BAD grammar ------------------------------------------------------ #
    grammar_str = """
    root   ::= "The animal is a " animal "."
    animal ::= "cat" | "fish" | "dog" E
    """

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

    # Our *inverse* processor
    bad_processor = BlockBadStateLogitsProcessor(grammar)

    prompts = [
        'The text says, "The animal is a dog." The answer is obvious. ',
        'I\'m going to say "The animal is a dog." Here I go! '
    ]
    input_ids = tokenizer(
        prompts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    ).to(device)["input_ids"]

    # ---------------------------------------------------------------------- #
    # Generate while avoiding any path that hits an "E" state               #
    # ---------------------------------------------------------------------- #
    outputs = model.generate(
        input_ids,
        max_length=30,
        logits_processor=[bad_processor],
        num_return_sequences=1,
    )

    for txt in tokenizer.batch_decode(outputs, skip_special_tokens=True):
        print(txt)
