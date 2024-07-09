import unittest

from transformers import GPT2TokenizerFast

from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


class GPT2TokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = GPT2TokenizerFast
    pretrained_name = "gpt2"

    def setUp(self):
        super().setUp()
