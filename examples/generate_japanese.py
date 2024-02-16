from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

import logging

logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-68m")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("JackFram/llama-68m")

    # Load grammar
    with open("examples/grammars/japanese.ebnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prefix1 = "Speak Japanese:"
    prefix2 = "My name is Jack, what is my name in Japanese?"
    input_ids = tokenizer(
        [prefix1, prefix2], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]

    output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=60,
        logits_processor=[grammar_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
    )
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generations)

    """
    'This is a valid json string for http request:{ "request": { "method": "GET", "headers": [], "content": "Content","type": "application" }}
    'This is a valid json string for shopping cart:This is a valid json string for shopping cart:{ "name": "MyCart", "price": 0, "value": 1 }
    """
