# 🤗 Transformers CFG

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 💭 Release news

### Latest release
#### **[v0.2.7 Latest](https://github.com/epfl-dlab/transformers-CFG/releases/tag/v0.2.7)** (2025-03-02)

#### **Features**

- **(CLI)** Types and MLX support (#93)
- **(Regex)** Negation, wildcard, and repetition bracket operators (#94, #95, #96, #104)
- **(Models)** Qwen2 and Qwen2.5 (#97)
- **(Logits)** Resuable `GrammarConstrainedLogitsProcessor` across generations for efficiency (#100)
- **(Backend)** Pytest for testing (#109)
- **(CI/CD)** GitHub Actions workflow for automation (#110)

#### **Bug fixes**

- Avoid computing full masks and optimized type additions (#101)
- Refactored grammar encoding to improve structure (#99)
- EOS token now correctly masks (#108)
- Multiple bugs removed and aesthetics improved (#107)

### Recent releases

- **[Gemma-2](https://github.com/epfl-dlab/transformers-CFG/pull/75)** — @fillassuncao (2024-08-16)
- **[DeepSeek](https://github.com/epfl-dlab/transformers-CFG/pull/73)** (2024-07-24)
- **LLaMA-3** (2024-07-08)
- **[JSON Schema](examples/grammars/custom_json_grammars/README.md)** (2024-05-13)
- **Token masking optimization** (2024-04-25)
- **[Phi](https://github.com/epfl-dlab/transformers-CFG/issues/34)** (2024-04-16)
- **[Online demo with JSON grammar](http://saibo-creator.xyz:7860/) at HF Space** (2024-04-10)
- **Unicode and multilingual grammar** (2024-02-29)
- **Text-Generation-WebUI** (2023-12-17)
  - We are pleased to announce that `transformers-cfg` has been integrated into the [Text-Generation-WebUI](https://github.com/oobabooga/text-generation-webui) project, allowing users to leverage CFG capabilities within this widely used text-generation interface ([Pull](https://github.com/oobabooga/text-generation-webui/pull/4953)).

## 🚀 Introduction

Initially developed as a pull request to the [Hugging Face Transformers](https://github.com/huggingface/transformers) library ([Pull](https://github.com/huggingface/transformers/pull/27557)), `transformers-cfg` extends the Hugging Face Transformers library to support constrained decoding through context-free grammars (CFG), offering a Transformers parellel for LlamaCPP's GBNF support, but with stricter generation rules.

## 💻 Installation

### Stable

Install the stable version via pip:

```bash
pip install transformers-cfg
```

### Development

For the latest updates, install directly from GitHub:

```bash
pip install git+https://github.com/epfl-dlab/transformers-CFG.git@main
```

## 🔧 Grammar quickstart
Let's set up a predictable generation method where the model would usually reply with "The animal is a dog." However, we'll force the model to say either "The animal is a cat" or "The animal is a fish," two other common domestic pets that contradict the inital text.

### Command-line interface (CLI)

The `transformers-cfg-cli` tool enables text generation using a model and a specified grammar. Unicode is supported.

```bash
transformers-cfg-cli generate \
    -m "microsoft/Phi-3-mini-4k-instruct" \
    -g "examples/grammars/json.ebnf" \
    -p "This is a valid JSON string for an HTTP request:" \
    --use_4bit \
    --max_new_tokens 60 \
    --repetition_penalty 1.1
# {"name":"John","age":30,"car":null}
```

Run `transformers-cfg-cli generate --help` for available options.

### Transformers *Torch*

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

if __name__ == "__main__":
    # Detect if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "facebook/opt-125m"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Define grammar string
    json_grammar = """

    root   ::= "The animal is a " animal "."

    animal ::= "cat" | "fish"

    """
    
    grammar = IncrementalGrammarConstraint(json_grammar, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prompts = [
        'The text says, "The animal is a dog." The answer is obvious. ', 'I\'m going to say "The animal is a dog." Here I go! '
              ]
    input_ids = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"].to(device)

    output = model.generate(
        input_ids,
        max_length=50,
        logits_processor=[grammar_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
    )
    
    # Decode output
    generations = tokenizer.batch_decode(output, skip_special_tokens=True)

    # Print all generations in for loop
    for generation in generations:
        print(generation)

```

#### Stream

<details>

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

if __name__ == "__main__":
    # Detect if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "facebook/opt-125m"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Define grammar as a string
    grammar_str = """

    root   ::= "The animal is a " animal "."

    animal ::= "cat" | "fish"

    """
    
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    prompts = [
        'The text says, "The animal is a dog." The answer is obvious. ', #'I\'m going to say "The animal is a dog." Here I go! '
              ]
    input_ids = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"].to(device)

    # Set up streaming
    streamer = TextStreamer(tokenizer)

    output = model.generate(
        input_ids,
        max_length=50,
        logits_processor=[grammar_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
        streamer=streamer
    )

```

</details>

### Transformers *Pipeline*

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

# Load model and tokenizer
model_id = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Detect if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# Define grammar string
json_grammar = """

root   ::= "The animal is a " animal "."

animal ::= "cat" | "fish"

"""

grammar = IncrementalGrammarConstraint(json_grammar, "root", tokenizer)
grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

# Initialize pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=100,
    batch_size=2,
)

# Generate text
generations = pipe(
    [
        'The text says, "The animal is a dog." The answer is obvious. ',
        'I\'m going to say "The animal is a dog." Here I go! '
    ],
    do_sample=False,
    logits_processor=[grammar_processor],
)

# Print results
for generation_group in generations:
    for generation in generation_group:
        print(generation['generated_text'])
```

### LlamaCPP Python

Coming soon!

## 💡 Why use `transformers-cfg`?

- **EBNF Grammar Support**: Uses Extended Backus-Naur Form (EBNF) for grammar description.
- **Seamless Integration**: Compatible with the llama-cpp project for easy replacement.
- **Broad Model Compatibility**: Works with all models in the 🤗 Transformers library.
- **Multilingual Grammar Support**: Enables grammars in various languages, including Chinese (中文), Japanese (日本語), Korean (한국어), Hindi (हिन्दी), Hebrew (עברית), Arabic (العربية), and emoji (🤗).  

## 🤔 What is a grammar?

Think of it as an enhanced version of regular expressions.

### Valid JSON object

```bnf
root ::= object
object ::= "{" pair ("," pair)* "}"
pair ::= string ":" value
string ::= '"' [a-zA-Z0-9]* '"'
value ::= string | object | "true" | "false" | "null"
```

For advanced grammar debugging, see our [debugging guide](docs/debugging_custom_grammars.md).

## 🛠 JSON schema

Learn to create grammars for complex JSON objects in our [documentation](examples/grammars/custom_json_grammars/README.md).

## 📜 Grammar collection

We maintain a collection of grammars in `examples/grammars`, aligned with llama-cpp:

- [json.ebnf](examples/grammars/json.ebnf): Valid JSON objects.
- [json_arr.ebnf](examples/grammars/json_arr.ebnf): Valid JSON arrays.
- [c.ebnf](examples/grammars/c.ebnf): Valid C programs.
- [chess.ebnf](examples/grammars/chess.ebnf): Valid chess moves.
- [arithmetic.ebnf](examples/grammars/arithmetic.ebnf): Valid arithmetic expressions.

## ✅ Supported models

- [Qwen2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f) and [2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
- [LLaMa](https://huggingface.co/baffo32/decapoda-research-llama-7B-hf) and Meta Llama
- [GPT-2](https://huggingface.co/openai-community/gpt2)
- [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Falcon](https://huggingface.co/tiiuae/falcon-7b)
- [OPT](https://huggingface.co/collections/facebook/opt-66ed00e15599f02966818844)

See [supported_models.yaml](docs/supported_models.yaml) for the full list whose extent is constantly being updated.

If you encounter an unsupported model, please open an issue or submit a pull request.

## 📖 Citation

If you find this work useful, please cite it with the reccomended citation:

```bibtex
@inproceedings{geng-etal-2023-grammar,
  title        = {Grammar-Constrained Decoding for Structured {NLP} Tasks without Finetuning},
  author       = {Geng, Saibo and Josifoski, Martin and Peyrard, Maxime and West, Robert},
  year         = 2023,
  month        = dec,
  booktitle    = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  publisher    = {Association for Computational Linguistics},
  address      = {Singapore},
  url          = {https://aclanthology.org/2023.emnlp-main.674},
  editor       = {Bouamor, Houda and Pino, Juan and Bali, Kalika}
}
```

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🙌 Acknowledgements

Derived from [torch-grammars](https://github.com/Shopify/torch-grammar), which was based on [llama-cpp](https://github.com/ggerganov/llama.cpp).
