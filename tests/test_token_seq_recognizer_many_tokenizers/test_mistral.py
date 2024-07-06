import unittest

from transformers import LlamaTokenizerFast

from tests._test_token_seq_recognizer_many_tokenizer_common import TokenizerTesterMixin

import logging


class MistralTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = LlamaTokenizerFast
    pretrained_name = "mistralai/Mistral-7B-v0.1"

    def setUp(self):
        super().setUp()
