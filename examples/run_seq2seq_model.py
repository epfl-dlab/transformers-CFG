"""
t5 tokenizer has a lot of unk tokens, such as open curly brace, close curly brace, tab, newline, etc.

"""
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers_cfg.grammar_utils import (
    IncrementalGrammarConstraint,
)
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

import logging

logging.basicConfig(level=logging.DEBUG)
transformers.logging.set_verbosity_debug()


if __name__ == "__main__":

    model_name = "facebook/bart-base"
    # model_name = "google-t5/t5-base"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # resize the embedding layer to match the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Load json grammar
    with open("examples/grammars/cIE.ebnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prefix1 = " entity1, relation1 , entity2 => "
    input_ids = tokenizer(
        [prefix1], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]

    output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=60,
        num_beams=1,
        logits_processor=[grammar_processor],
        num_return_sequences=1,
    )
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generations)
