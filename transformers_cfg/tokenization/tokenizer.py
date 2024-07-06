class Tokenizer:
    def __init__(self, hf_tokenizer):
        self.hf_tokenizer = hf_tokenizer

    def real_vocab_size(self):
        # for codegen and phi tokenizer, there are a few tokens which are empty and reserved for future use
        if "codegen" in self.hf_tokenizer.__class__.__name__.lower():
            return 50257  # 50 k tokens + 256 for bytes + 1 for EOS
        else:
            return len(self.hf_tokenizer.vocab)
