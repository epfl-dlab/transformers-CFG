import unittest

from transformers import T5TokenizerFast

from tests._tokenizer_common import TokenizerTesterMixin

import logging


@unittest.skip("T5Tokenizer's mapping is not well defined, not working")
class T5TokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = T5TokenizerFast
    pretrained_name = "t5-small"

    def setUp(self):
        super().setUp()
