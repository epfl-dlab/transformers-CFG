from transformers import (
    GPT2TokenizerFast,
    BartTokenizerFast,
    LlamaTokenizerFast,
    T5TokenizerFast,
    CodeGenTokenizerFast,
    PreTrainedTokenizerFast,
    GemmaTokenizerFast
)

SUPPORTED_TOKENIZERS = {
    GPT2TokenizerFast,
    BartTokenizerFast,
    LlamaTokenizerFast,
    T5TokenizerFast,
    CodeGenTokenizerFast,
    PreTrainedTokenizerFast,
    GemmaTokenizerFast
}
