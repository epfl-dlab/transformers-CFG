from transformers import AutoTokenizer
from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


class TestGemma2Tokenizer(TokenizerTesterMixin):
    tokenizer_class = AutoTokenizer
    pretrained_name = "Transformers-CFG/gemma-2-2b-it-tokenizer"

    def setup(self):
        self.setup_tokenizer()
