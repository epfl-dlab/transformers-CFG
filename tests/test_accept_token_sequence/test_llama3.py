from transformers import AutoTokenizer
from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


class TestLlama3Tokenizer(TokenizerTesterMixin):

    tokenizer_class = AutoTokenizer
    pretrained_name = "Transformers-CFG/Meta-Llama-3-8B-tokenizer"

    def setup(self):
        self.setup_tokenizer()
