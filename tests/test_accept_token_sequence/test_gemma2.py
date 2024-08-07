import unittest

from transformers import AutoTokenizer

from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


class Gemma2TokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = AutoTokenizer
    pretrained_name = "google/gemma-2-2b-it"

    def setUp(self):
        super().setUp()
