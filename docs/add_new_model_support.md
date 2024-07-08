# Add new model support

In case you want to use a new model that is not supported yet, here is a guide to add support for it.

In the following guide, we will use the newly released `meta-llama/Meta-Llama-3-8B` model as an example.


### Step 1: Check if the model is supported

`transformers-cfg-cli` is a command-line tool that can be used to check if a model is supported by `transformers-cfg`.

```bash
transformers-cfg-cli check meta-llama/Meta-Llama-3-8B
# Model meta-llama/Meta-Llama-3-8B is not supported.
# OR
# Model meta-llama/Meta-Llama-3-8B is supported.
```




### Step 1: Find the underlying tokenizer class


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

print(tokenizer.__class__)
# transformers.tokenization_utils_fast.PreTrainedTokenizerFast
```

As you can see here, the tokenizer class is `PreTrainedTokenizerFast`.

There are several caveats to this:


1. Many models can share the same tokenizer class, even though HF sometimes make wrapper classes to make the tokenizer class more user-friendly.

For example, both `mistralai/Mistral-7B-v0.1` and `meta-llama/Llama-2-7b-hf` use `LlamaTokenizerFast` as their tokenizer class.

```python
from transformers import AutoTokenizer

mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

print(mistral_tokenizer.__class__)
# transformers.models.llama.tokenization_llama.LlamaTokenizerFast

print(llama_tokenizer.__class__)
# transformers.models.llama.tokenization_llama.LlamaTokenizerFast
```

2. Two models in the same family but different generations can have different tokenizer classes.

For example, `meta-llama/Meta-Llama-3-8B` uses `PreTrainedTokenizerFast` while `meta-llama/Llama-2-7b-hf` uses `LlamaTokenizerFast`.



### Step 2: See if the tokenizer class is already supported
