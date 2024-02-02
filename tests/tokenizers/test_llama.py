import unittest

from transformers import LlamaTokenizer

from tests.tokenizer_common import TokenizerTesterMixin

import logging
logging.basicConfig(level=logging.DEBUG)


class GPT2TokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = LlamaTokenizer
    pretrained_name = "saibo/llama-1B"

    def setUp(self):
        super().setUp()

