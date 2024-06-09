import unittest

from transformers import GPT2TokenizerFast
from tests._tokenizer_common import TokenizerTesterMixin

import logging


class Llama3TokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = GPT2TokenizerFast
    pretrained_name = "meta-llama/Meta-Llama-3-8B"

    def setUp(self):
        super().setUp()
