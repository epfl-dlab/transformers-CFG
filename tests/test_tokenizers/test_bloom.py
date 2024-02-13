import unittest

from transformers import BloomTokenizerFast

from tests._tokenizer_common import TokenizerTesterMixin

import logging


class BloomTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BloomTokenizerFast
    pretrained_name = "bigscience/bloom-560m"

    def setUp(self):
        super().setUp()
