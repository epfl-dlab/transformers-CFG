# 🤗 Transformers CFG

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 💭 Latest News

- **[Gemma-2 Support](https://github.com/epfl-dlab/transformers-CFG/pull/75)** — Thanks to @fillassuncao (2024-08-16)
- **[DeepSeek Support](https://github.com/epfl-dlab/transformers-CFG/pull/73)** (2024-07-24)
- **LLAMA-3 Support** (2024-07-08)
- **[JSON Schema as Constraint Support](examples%2Fgrammars%2Fcustom_json_grammars%2FREADME.md)** (2024-05-13)
- **[Token Masking Optimization](#efficiency)** (2024-04-25)
- **[Phi Support](https://github.com/epfl-dlab/transformers-CFG/issues/34)** (2024-04-16)
- **[Online Demo with JSON Grammar](http://saibo-creator.xyz:7860/) at HF Space** (2024-04-10)
- **Unicode (Multilingual) Grammar Support** (2024-02-29)
- **Integration with Text-Generation-WebUI** (2023-12-17)

We are thrilled to announce that `transformers-cfg` has been integrated into the [Text-Generation-WebUI](https://github.com/oobabooga/text-generation-webui) project, enabling users to utilize our CFG capabilities within this popular web interface for text generation. For more details, see the [relevant pull request](https://github.com/oobabooga/text-generation-webui/pull/4953).

## 🚀 Introduction

`transformers-cfg` is an extension library for the popular Transformers library by Hugging Face, tailored for working with context-free grammars (CFG). This package provides additional tools and functionalities to enhance your experience with natural language processing tasks involving CFGs.

Initially developed as a pull request to the [Hugging Face Transformers](https://github.com/huggingface/transformers) library, you can find the relevant discussion [here](https://github.com/huggingface/transformers/pull/27557).

## 💻 Installation

- **Stable Version:**

  Install the stable version of `transformers-cfg` using pip:

  ```bash
  pip install transformers-cfg
  ```

- **Development Version:**

  For the latest code and updates, install directly from the GitHub repository:

  ```bash
  pip install git+https://github.com/epfl-dlab/transformers-CFG.git@main
  ```

  This installs the package from the `main` branch.

## 🔧 Quick Start: Force LLM to Generate a Valid JSON Object

### Command-Line Interface

`transformers-cfg-cli` is a command-line tool that allows you to generate text using a model and a grammar. You can specify the model, grammar, prompts, and other parameters to generate text that conforms to the specified grammar.

```bash
transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/json.ebnf" \
    -p "This is a valid json string for http request:" \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1
# {"name":"John","age":30,"car":null}
```

We support also Unicode characters in the grammar:

```bash
transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/chinese.ebnf" \
    -p "Translate the following sentence into Chinese: My neighbor is a very nice person. -> " \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1
```

`transformers-cfg-cli generate --help` provides a list of available options and arguments.


<details>
<summary>Click here to see an example of generating a JSON object with minimal changes to HF code, or check it out in <code>examples/generate_json.py</code></summary>

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

if __name__ == "__main__":
    # Detect if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "mistralai/Mistral-7B-v0.1"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Load JSON grammar
    with open("examples/grammars/json.ebnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prompts = ["This is a valid json string for http request:", "This is a valid json string for shopping cart:"]
    input_ids = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]

    output = model.generate(
        input_ids,
        max_length=50,
        logits_processor=[grammar_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
    )
    # Decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(generations)

    """
    'This is a valid json string for http request:{ "request": { "method": "GET", "headers": [], "content": "Content","type": "application" }}'
    'This is a valid json string for shopping cart:{ "name": "MyCart", "price": 0, "value": 1 }'
    """
```

</details>

<details>
<summary>Click here to see an example with HF pipeline API, or check it out in <code>examples/pipeline_json.py</code></summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# Load grammar
with open(f"examples/grammars/json.ebnf", "r") as file:
    grammar_str = file.read()
grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

# Initialize pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_length=50,
    batch_size=2,
)

generations = pipe(
    [
        "This is a valid json string for http request: ",
        "This is a valid json string for shopping cart: ",
    ],
    do_sample=False,
    logits_processor=[grammar_processor],
)
```

</details>

## 💡 Why Should I Use `transformers-cfg`?

- **EBNF Grammar Support:** We support the Extended Backus-Naur Form (EBNF) for grammar description.
- **Seamless Integration:** Our grammar interface is compatible with the llama-cpp project, allowing you to replace llama-cpp with `transformers-cfg` easily.
- **Model Compatibility:** Use any model from the 🤗 Transformers library, including those not supported by llama-cpp.
- **Multilingual Grammar Support:** We support grammars in multiple languages, allowing you to use characters from various languages, including 中文, 日本語, 한국어, हिन्दी, العربية, עברית, and emoji 🤗.

## 🤔 What Is a Grammar?

TL;DR: Think of it as an enhanced version of regular expressions.

<details>
<summary>Here is a simple example of a JSON grammar:</summary>

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

This grammar describes the structure of a JSON object. It specifies that a JSON object consists of key-value pairs, where the key is a string, and the value can be a string, another object, or a boolean value.

You can use grammars to describe simple but useful constructs, such as valid email addresses, URLs, or phone numbers:

```
phone_number ::= "+" [0-9]+
```

</details>


For advanced grammar debugging, check out our [debugging guide](docs/debugging_custom_grammars.md).

## Automatic JSON Schema Grammar Conversion[Experimental]

Learn how to automatically create custom grammars for complex JSON objects in our [documentation](examples%2Fgrammars%2Fcustom_json_grammars%2FREADME.md) on JSON schema to grammar conversion.

## Grammar Collection

We provide a collection of grammars in the `examples/grammars` folder, which are mostly identical to the grammars in the llama-cpp project. We try to keep these grammars up-to-date with the original project, though we cannot yet guarantee that all grammars from llama-cpp can be directly used in `transformers-cfg`.

Available grammars include:

- [json.ebnf](examples%2Fgrammars%2Fjson.ebnf): For generating valid JSON objects.
- [json_arr.ebnf](examples%2Fgrammars%2Fjson_arr.ebnf): For generating valid JSON arrays.
- [c.ebnf](examples%2Fgrammars%2Fc.ebnf): For generating valid C programs.
- [chess.ebnf](examples%2Fgrammars%2Fchess.ebnf): For generating valid chess moves.
- [arithmetic.ebnf](examples%2Fgrammars%2Farithmetic.ebnf): For generating valid arithmetic expressions.

## Supported Models

- [LLaMa Family Models](https://huggingface.co/baffo32/decapoda-research-llama-7B-hf)
- [GPT Family Models](https://huggingface.co/openai-community/gpt2)
- [Bloom Family Models](https://huggingface.co/bigscience/bloom)
- [Mistral Family Models](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Falcon Family Models](https://huggingface.co/tiiuae/falcon-7b)
- ...

See [supported_models.yaml](docs%2Fsupported_models.yaml) for the full list of supported models.

As a rule of thumb, all models with the same tokenizer should be naturally supported.

If you find any model that is not supported, please open an issue or submit a pull request.


## Citation

**Please consider citing our work if you find the provided resources useful:**

```bibtex
@inproceedings{geng-etal-2023-grammar,
	title        = {Grammar-Constrained Decoding for Structured {NLP} Tasks without Finetuning},
	author       = {Geng, Saibo  and Josifoski, Martin  and Peyrard, Maxime  and West, Robert},
	year         = 2023,
	month        = dec,
	booktitle    = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
	publisher    = {Association for Computational Linguistics},
	address      = {Singapore},
	url          = {https://aclanthology.org/2023.emnlp-main.674},
	editor       = {Bouamor, Houda  and Pino, Juan  and Bali, Kalika}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

This project is derived from the [torch-grammars](https://github.com/Shopify/torch-grammar) project, which was itself derived from the [llama-cpp](https://github.com/ggerganov/llama.cpp) project.
