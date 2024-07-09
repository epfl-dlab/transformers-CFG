import unittest

from transformers import LlamaTokenizerFast

from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


class MistralTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = LlamaTokenizerFast
    pretrained_name = "mistralai/Mistral-7B-v0.1"

    def setUp(self):
        super().setUp()
