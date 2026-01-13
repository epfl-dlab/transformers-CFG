from typing import List
from transformers import AutoTokenizer


whisper_tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny.en")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

from transformers_cfg.tokenization.mapping import Token2ByteMapping
from transformers_cfg.tokenization import ByteTrie, TCFG_Tokenizer


mapping = Token2ByteMapping.from_hf_tokenizer(whisper_tokenizer)

print(mapping)

tcfg_tokenizer = TCFG_Tokenizer.from_hf_tokenizer(whisper_tokenizer)
print(tcfg_tokenizer)

print(tcfg_tokenizer.real_vocab_size())
print(tcfg_tokenizer.special_token_ids)

byte_trie = ByteTrie.from_tokenizer(whisper_tokenizer)

# byte_trie.visualize(max_depth=2)

whisper_vocab = whisper_tokenizer.get_vocab()

gpt2_vocab = gpt2_tokenizer.get_vocab()

print(f"Whisper vocab size: {len(whisper_vocab)}")  # 51864
print(f"GPT2 vocab size: {len(gpt2_vocab)}")  # 50257

# check if all gpt2 vocab is in whisper vocab

whisper_token_set = set(whisper_vocab.keys())
gpt2_token_set = set(gpt2_vocab.keys())

print(
    f"All gpt2 vocab in whisper vocab: {gpt2_token_set.issubset(whisper_token_set)}"
)  # True

# print the tokens that are not in gpt2 vocab
whisper_only_tokens = whisper_token_set - gpt2_token_set
print(f"Whisper only tokens: {sorted(whisper_only_tokens)}")  # ['


# timestamp token ids

timestamp_token_ids: List[int] = whisper_tokenizer.timestamp_ids()

print(f"Number of timestamp token ids: {len(timestamp_token_ids)}")  # 0

print(f"Number of special token ids: {len(whisper_tokenizer.all_special_ids)}")  # 0
