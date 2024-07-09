import unittest

from transformers import LlamaTokenizerFast

from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


class LlamaTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = LlamaTokenizerFast
    pretrained_name = "saibo/llama-1B"

    def setUp(self):
        super().setUp()
