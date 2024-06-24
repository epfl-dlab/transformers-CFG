import unittest

from transformers import BloomTokenizerFast

from tests._test_token_seq_recognizer_many_tokenizer_common import TokenizerTesterMixin

import logging


@unittest.skip("Bloom is not supported and will be removed")
class BloomTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BloomTokenizerFast
    pretrained_name = "bigscience/bloom-560m"

    def setUp(self):
        super().setUp()
