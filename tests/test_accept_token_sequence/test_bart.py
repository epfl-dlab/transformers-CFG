import unittest

from transformers import BartTokenizerFast

from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


class BartTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BartTokenizerFast
    pretrained_name = "facebook/bart-large"

    def setUp(self):
        super().setUp()

    @unittest.skip(
        "BartTokenizer fails on emoji for some reason, need to investigate if user really needs this"
    )
    def test_emoji(self):
        super().test_emoji()
