import unittest

from transformers import GPT2TokenizerFast

from tests._tokenizer_common import TokenizerTesterMixin

import logging


class GPT2TokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = GPT2TokenizerFast
    pretrained_name = "gpt2"

    def setUp(self):
        super().setUp()
