import unittest

from transformers import GPT2Tokenizer

from tests.tokenizer_common import TokenizerTesterMixin

import logging
logging.basicConfig(level=logging.DEBUG)


class GPT2TokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = GPT2Tokenizer
    pretrained_name = "gpt2"

    def setUp(self):
        super().setUp()

