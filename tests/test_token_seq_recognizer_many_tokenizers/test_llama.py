import unittest

from transformers import LlamaTokenizerFast

from tests._test_token_seq_recognizer_many_tokenizer_common import TokenizerTesterMixin

import logging


class LlamaTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = LlamaTokenizerFast
    pretrained_name = "saibo/llama-1B"

    def setUp(self):
        super().setUp()
