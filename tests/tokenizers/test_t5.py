import unittest

from transformers import T5Tokenizer

from tests.tokenizer_common import TokenizerTesterMixin

import logging
logging.basicConfig(level=logging.DEBUG)


class GPT2TokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = T5Tokenizer
    pretrained_name = "t5-small"

    def setUp(self):
        super().setUp()

