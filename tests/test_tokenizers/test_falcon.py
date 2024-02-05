import unittest

from transformers import PreTrainedTokenizer, AutoTokenizer

from tests._tokenizer_common import TokenizerTesterMixin

import logging

logging.basicConfig(level=logging.DEBUG)


class FalconTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = AutoTokenizer
    pretrained_name = "tiiuae/falcon-7b"

    def setUp(self):
        super().setUp()
