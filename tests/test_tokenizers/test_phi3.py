import unittest

from transformers import T5TokenizerFast

from tests._tokenizer_common import TokenizerTesterMixin

import logging


class Phi3TokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = T5TokenizerFast
    pretrained_name = "microsoft/Phi-3-mini-4k-instruct"

    def setUp(self):
        super().setUp()
