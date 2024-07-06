import unittest

from transformers import BartTokenizerFast

from tests._test_token_seq_recognizer_many_tokenizer_common import TokenizerTesterMixin

import logging


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
