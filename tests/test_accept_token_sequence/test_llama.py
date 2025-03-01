from transformers import LlamaTokenizerFast
from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


class TestLlamaTokenizer(TokenizerTesterMixin):
    tokenizer_class = LlamaTokenizerFast
    pretrained_name = "saibo/llama-1B"

    def setup(self):
        self.setup_tokenizer()
