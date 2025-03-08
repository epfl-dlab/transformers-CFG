import io
import torch
import logging
from contextlib import redirect_stderr
from llama_cpp import Llama
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

# Define your EBNF grammar (you can replace this with your own)
ebnf_grammar = """

    root   ::= "The animal is a " animal "."

    animal ::= "cat" | "fish"

    """

# Load the tokenizer matching your model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5b")

# Redirect stderr and load the model via llama-cpp-python
f = io.StringIO()
with redirect_stderr(f):
    model = Llama(model_path="qwen2.5-1.5b-q8_0.gguf", n_ctx=8000, verbose=False)

# Create the grammar constraint and the logits processor with the new parameter.
grammar_constraint = IncrementalGrammarConstraint(ebnf_grammar, "root", tokenizer)
grammar_processor = GrammarConstrainedLogitsProcessor(grammar_constraint, adapter="llama-cpp-python")

# Define a prompt.
prompt = """The text says, "The animal is a dog." The answer is obvious. """

# Use the text completion API with the logits processor.
response = model.create_completion(
    stream=True,
    prompt=prompt,
    logits_processor=[grammar_processor],
    max_tokens=100,
)

for token in response:
    token_text = token["choices"][0]["text"]
    print(token_text, end="", flush=True)
