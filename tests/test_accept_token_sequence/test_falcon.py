import unittest

from transformers import AutoTokenizer

from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


@unittest.skip("Falcom is not supported and will be removed")
class FalconTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = AutoTokenizer
    pretrained_name = "tiiuae/falcon-7b"

    def setUp(self):
        super().setUp()
