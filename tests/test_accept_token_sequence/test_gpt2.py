from transformers import GPT2TokenizerFast
from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


class TestGPT2Tokenizer(TokenizerTesterMixin):
    tokenizer_class = GPT2TokenizerFast
    pretrained_name = "gpt2"

    def setup(self):
        self.setup_tokenizer()
