import cProfile
import pstats
import io

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
    with open("examples/grammars/japanese.ebnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prefix1 = "English: coffee, Japanese: "
    prefix2 = "English: dog, Japanese: "
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
    print(output)
    print(generations)


if __name__ == "__main__":
    # Profiling
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    s = io.StringIO()
    sortby = "tottime"  # Sort by total time
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats()
    profiling_results = s.getvalue()
    with open("profiling_japanese.txt", "w") as f:
        f.write(profiling_results)

    from transformers_cfg.token_grammar_recognizer import AbsTokenGrammarRecognizer

    print(AbsTokenGrammarRecognizer.token_acceptance_for_stack.cache_info())
