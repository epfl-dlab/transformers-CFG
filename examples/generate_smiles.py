import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.recognizer import StringRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.parser import parse_ebnf


if __name__ == "__main__":

    model_id = "mistralai/Mistral-7B-v0.1"

    # Detect if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # Load model to defined device
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device) 

    # Load grammar
    for grammar_name in ['generic', 'isocyanates', 'acrylates', 'chain_extenders']:
        with open(f"examples/grammars/SMILES/{grammar_name}.ebnf", "r") as file:
            grammar_str = file.read()

        parsed_grammar = parse_ebnf(grammar_str)
        first_rule = grammar_str.split('\n')[0]
        print(f"{grammar_name}: {first_rule}")
        grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

        # Generate
        prefix1 = f"This is a {grammar_name} SMILES string:"

        input_ids = tokenizer(
            [prefix1], add_special_tokens=False, return_tensors="pt", padding=True
        )["input_ids"].to(
            device
        )  # Move input_ids to the same device as model

        max_new_tokens = 20
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
        for generation, gen_type in zip(generations, ['unconstrained:', 'constrained:']):
            print("_" * 10)
            print(gen_type)
            print(generation)
            print("_" * 10)
