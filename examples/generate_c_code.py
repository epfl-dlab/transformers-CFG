import cProfile
import pstats
import io
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

############################################################
#
# use llama to generate C code
#
############################################################


def main():
    global device, tokenizer, tokenizer
    # Detect if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2-large").to(
        device
    )  # Load model to defined device
    # Load grammar
    with open("examples/grammars/c.ebnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
    # Generate
    prefix1 = "#include <stdio.h>\n"
    input_ids = tokenizer(
        [prefix1], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"].to(
        device
    )  # Move input_ids to the same device as model
    output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=20,
        logits_processor=[grammar_processor],
        repetition_penalty=3.0,
        num_return_sequences=1,
    )
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generations)
    print(output)
    """
    #include <stdio.h>
    int thresh_f(int n){return (1-threshold);}
    """


if __name__ == "__main__":
    main()
