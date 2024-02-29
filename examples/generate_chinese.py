from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

import logging

logging.basicConfig(level=logging.DEBUG)


def main():

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # JackFram/llama-68m"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")  # Load model to defined device

    # Load grammar
    with open("examples/grammars/chinese.ebnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer, unicode=True)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prefix1 = "English: coffee, Chinese: "
    prefix2 = "English: dog, Chinese: "
    input_ids = tokenizer(
        [prefix1, prefix2], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]

    output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=20,
        logits_processor=[grammar_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
    )
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)

    print(generations)


if __name__ == "__main__":
    main()
