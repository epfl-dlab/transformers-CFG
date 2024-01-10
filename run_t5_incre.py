import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers_cfg.grammar_utils import (
    IncrementalGrammarConstraint,
    VanillaGrammarConstraint,
)
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

import logging

logging.basicConfig(level=logging.DEBUG)
transformers.logging.set_verbosity_debug()


if __name__ == "__main__":

    model_name = "t5-small"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # resize the embedding layer to match the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Load json grammar
    with open("examples/grammars/grammar_t5_fail_1.ebnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar, parse_start_index=1)

    # Generate
    prefix1 = "This is a valid json string for http request:"
    prefix2 = "This is a valid json string for shopping cart:"
    input_ids = tokenizer(
        [prefix1, prefix2], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]

    output = model.generate(
        input_ids,
        do_sample=False,
        max_length=10,
        num_beams=1,
        logits_processor=[grammar_processor],
        repetition_penalty=5.0,
        num_return_sequences=1,
    )
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generations)

    """
    'This is a valid json string for http request:{ "request": { "method": "GET", "headers": [], "content": "Content","type": "application" }}
    'This is a valid json string for shopping cart:This is a valid json string for shopping cart:{ "name": "MyCart", "price": 0, "value": 1 }
    """
