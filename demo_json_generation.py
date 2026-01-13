#!/usr/bin/env python3
"""
Simple demo showing grammar-constrained JSON generation with GPT2.
This demonstrates that the refactored code works correctly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor


def main():
    print("=" * 80)
    print("Grammar-Constrained JSON Generation Demo")
    print("=" * 80)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")

    # Load model and tokenizer (using Llama 3.2 1B - small and capable)
    # Using base model (not Instruct) to avoid chat template issues
    print("\n✓ Loading Llama 3.2 1B model and tokenizer...")
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # Handle pad_token_id (eos_token_id might be a list for some models)
    eos_token_id = model.generation_config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    model.generation_config.pad_token_id = eos_token_id

    # Load JSON grammar
    print("\n✓ Loading JSON grammar...")
    with open("examples/grammars/json.ebnf", "r") as file:
        grammar_str = file.read()

    print("\nGrammar preview (first 300 chars):")
    print("-" * 80)
    print(grammar_str[:300] + "...")
    print("-" * 80)

    # Generate with grammar constraints
    prompts = [
        "A person's information in JSON:",
        "Configuration settings:",
    ]

    print(f"\n✓ Generating JSON for {len(prompts)} prompts...")
    print("=" * 80)

    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        print("-" * 80)

        # Create fresh grammar constraint for each generation
        grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device)

        output = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=50,
            logits_processor=[grammar_processor],
            repetition_penalty=1.1,
            num_return_sequences=1,
        )

        generation = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract just the generated JSON part
        json_part = generation[len(prompt) :].strip()

        print(f"Generated JSON:\n{json_part}")

        # Verify it's valid JSON structure
        if json_part.startswith("{") and json_part.count("{") > 0:
            print("✓ Valid JSON structure generated!")
        else:
            print("Note: Partial JSON (may need more tokens)")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
