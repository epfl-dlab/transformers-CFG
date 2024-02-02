# 🤗 Transformers CFG

## Latest News

**Integration with Text-Generation-WebUI** (2023-12-17)

We are thrilled to announce that `transformers_cfg` has been used in the [Text-Generation-WebUI](https://github.com/oobabooga/text-generation-webui) project.
This integration enables users to utilize our CFG capabilities within the popular, 30.5K-starred web interface for text generation.
For more details, see [Relevent Pull Request](https://github.com/oobabooga/text-generation-webui/pull/4953)


## Introduction
`transformers_cfg` is an extension library for the popular Transformers library by Hugging Face, tailored for working with context-free grammars (CFG).
This package provides additional tools and functionalities to enhance your experience with natural language processing tasks involving CFGs.

It was initially developed as a pull request to the [Hugging Face Transformers](https://github.com/huggingface/transformers) library.
See relevant discussion [here](https://github.com/huggingface/transformers/pull/27557).

## Installation

```bash
pip install git+https://github.com/epfl-dlab/transformers-CFG.git
```

## QuickStart: Force LLM to generate a valid json object

The below example can be found in `examples/generate_json.py`

```python

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Load json grammar
    with open("examples/grammars/json.ebnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prefix1 = "This is a valid json string for http request:"
    prefix2 = "This is a valid json string for shopping cart:"
    input_ids = tokenizer([prefix1, prefix2], add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]

    output = model.generate(
        input_ids,
        do_sample=False,
        max_length=50,
        num_beams=2,
        logits_processor=[grammar_processor],
        repetition_penalty=5.0,
        num_return_sequences=1,
    )
    # decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generations)

    """
    'This is a valid json string for http request:{ "request": { "method": "GET", "headers": [], "content": "Content","type": "application" }}
    'This is a valid json string for shopping cart:{ "name": "MyCart", "price": 0, "value": 1 }
    """

```

## Why should I use transformers-CFG?

- We offer the same grammar interface as llama-cpp project, allowing you to drop-in replace llama-cpp with transformers-CFG.
- We allow you to use any of the models in the 🤗 Transformers library, including the ones that are not supported by llama-cpp.

## What is grammar ?

TL;DR: Think of it as an enhanced version of regular expressions. 

Here is an example of a simplified JSON grammar:
```bnf
# A JSON object is the root of the grammar
root ::= object 

# An object starts with "{" and ends with "}" and contains pairs separated by ","
object ::= "{" pair ("," pair)* "}" 

# A pair is a string followed by a ":" and a value
pair ::= string ":" value

# A string is a sequence of alphanumeric characters enclosed in double quotes
string ::= '"' [a-zA-Z0-9]* '"' 

# A value can be a string, another object, or a boolean value
value ::= string | object | "true" | "false" | "null"
```

This grammar describes the structure of a JSON object. It specifies that a JSON object is a pair of key-value pairs, where the key is a string and the value can be a string, another object, or a boolean value.

Grammar doesn't need to be complicated.
You can use it to describe very simple but useful things, like a valid email address, a valid URL, or phone number.
```
phone_number ::= "+" [0-9]+
```

More details can be found in this [doc from llama-cpp](https://github.com/ggerganov/llama.cpp/tree/master/grammars)

### Automatic Grammar Generation
Here is an awesome tool to generate grammars for you: [Grammar Builder](https://grammar.intrinsiclabs.ai/)

### Grammar Collection

We provide a collection of grammars in the `examples/grammars` folder, which are mostly identical to the grammars in llama-cpp project.
We try to keep the grammars up-to-date with the original grammars from llama-cpp project.
But up to now, we can not yet guarantee that all grammars from llama-cpp project can be directly used in transformers-CFG.

The list of grammars contains:
- [json.ebnf](examples%2Fgrammars%2Fjson.ebnf): A grammar for generating valid json objects.
- [json_arr.ebnf](examples%2Fgrammars%2Fjson_arr.ebnf): A grammar for generating valid json arrays.
- [c.ebnf](examples%2Fgrammars%2Fc.ebnf): A grammar for generating valid C programs.
- [chess.ebnf](examples%2Fgrammars%2Fchess.ebnf): A grammar for generating valid chess moves.
- [arithmetic.ebnf](examples%2Fgrammars%2Farithmetic.ebnf): A grammar for generating valid arithmetic expressions.


## Supported Models

`Transformers-CFG` is tokenizer-dependent and model-agnostic.
All models with the same tokenizer should naturally be supported.

If you find any model that is not supported, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgement

This project is derived from the [torch-grammars](https://github.com/Shopify/torch-grammar) project, which was derived from the [llama-cpp](https://github.com/ggerganov/llama.cpp) project.
