import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.parser import parse_ebnf
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PDDL strings")
    parser.add_argument(
        "--model-id",
        type=str,
        default="/dlabdata1/llm_hub/Mistral-7B-v0.1",
        help="Model ID",
    )
    parser.add_argument("--device", type=str, help="Device to put the model on")
    parser.add_argument(
        "--pddl-type",
        type=str,
        choices=["blocks", "depot", "satellite", "depot_typed", "satellite_typed"],
        default="blocks",
        help="Type of PDDL to generate",
    )
    return parser.parse_args()


one_shot_prompts = {
    "blocks": "(put-down a) (unstack-and-stack c b d) (pick-up-and-stack b c)",
    "depot": "(drive truck0 depot0 distributor0) (lift-and-drive truck0 hoist0 crate0 pallet0 depot0 depot0) (lift hoist2 crate2 crate1 distributor1)",
    "satellite": "(switch-on instrument1 satellite3) (turn-to satellite1 direction4 direction0)",
}


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
    with open(f"examples/grammars/PDDL/{args.pddl_type}.ebnf", "r") as file:
        grammar_str = file.read()

    parsed_grammar = parse_ebnf(grammar_str)
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    pddl_domain = args.pddl_type.split("_")[0]
    prompts = [
        f"Give me two examples of the {pddl_domain} command sequence:\n"
        + f"1. {one_shot_prompts[pddl_domain]}\n2. "
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

    parsed_grammar = parse_ebnf(grammar_str)
    string_grammar = StringRecognizer(
        parsed_grammar.grammar_encoding, parsed_grammar.symbol_table["root"]
    )

    # decode outputs (possibly of different lengths across decoding modes)
    generations = tokenizer.batch_decode(
        unconstrained_output, skip_special_tokens=True
    ) + tokenizer.batch_decode(constrained_output, skip_special_tokens=True)
    print()

    for i in range(n_examples):
        unconstrained_generation = generations[i]
        constrained_generation = generations[i + n_examples]
        prompt = prompts[i]

        for generation, generation_type in zip(
            [unconstrained_generation, constrained_generation],
            ["unconstrained", "constrained"],
        ):
            print(f"The {generation_type} generation:\n{generation}")
            print(
                f"The {generation_type} generation is a valid prefix for the grammar: {string_grammar._accept_prefix(generation[len(prompt):])}"
            )
            print(
                f"The {generation_type} generation is a valid sentence for the grammar: {string_grammar._accept_string(generation[len(prompt):])}"
            )
            print()


if __name__ == "__main__":
    main()


##########################
# Example output:
#
# BLOCKS:
#
# The unconstrained generation:
# Give me two examples of the blocks command sequence:
# 1. (put-down a) (unstack-and-stack c b d) (pick-up-and-stack b c)
# 2. 30,45,(move left),(turn right)(go forward). The first example is an action that can be performed by any robot with three arms and four objects in its workspace; it does not depend on what those particular arm/object
# The unconstrained generation is a valid prefix for the grammar: False
# The unconstrained generation is a valid sentence for the grammar: False

# The constrained generation:
# Give me two examples of the blocks command sequence:
# 1. (put-down a) (unstack-and-stack c b d) (pick-up-and-stack b c)
# 2. (pick-up e) (put-down e) (unstack-and-stack b c d) (put-down a) (unstack-and-stack c b
# The constrained generation is a valid prefix for the grammar: True
# The constrained generation is a valid sentence for the grammar: False
#
# DEPOT:
# The unconstrained generation:
# Give me two examples of the depot command sequence:
# 1. (drive truck0 depot0 distributor0) (lift-and-drive truck0 hoist0 crate0 pallet0 depot0 depot0) (lift hoist2 crate2 crate1 distributor1)
# 2. 3567894(move robot arm to position A)(pick up object from table and place it on shelf B). The first example is a simple one, where we have three trucks that are going back into their respective parking spots after
# The unconstrained generation is a valid prefix for the grammar: False
# The unconstrained generation is a valid sentence for the grammar: False

# The constrained generation:
# Give me two examples of the depot command sequence:
# 1. (drive truck0 depot0 distributor0) (lift-and-drive truck0 hoist0 crate0 pallet0 depot0 depot0) (lift hoist2 crate2 crate1 distributor1)
# 2. (load crate3 crate4 distributor1 distributor1) (unload truck0 pallet5 depot0 distributor1) (drop truck
# The constrained generation is a valid prefix for the grammar: True
# The constrained generation is a valid sentence for the grammar: False

# SATELLITE:
# The unconstrained generation:
# Give me two examples of the satellite command sequence:
# 1. (switch-on instrument1 satellite3) (turn-to satellite1 direction4 direction0)
# 2. ......(move to position5 distance6 angle7 )... The first example is a simple one, but it shows how we can use satellites as instruments and also move them around in space using their own commands for movement or orientation change etc.,
# The unconstrained generation is a valid prefix for the grammar: False
# The unconstrained generation is a valid sentence for the grammar: False

# The constrained generation:
# Give me two examples of the satellite command sequence:
# 1. (switch-on instrument1 satellite3) (turn-to satellite1 direction4 direction0)
# 2. (take-image satellite1 instrument5 direction0 direction1) (calibrate instrument5 satellite1 direction0) (switch-off instrument5 satell
# The constrained generation is a valid prefix for the grammar: True
# The constrained generation is a valid sentence for the grammar: False

##########################
