import unittest

from transformers import BloomTokenizerFast

from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


@unittest.skip("Bloom is not supported and will be removed")
class BloomTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BloomTokenizerFast
    pretrained_name = "bigscience/bloom-560m"

    def setUp(self):
        super().setUp()
