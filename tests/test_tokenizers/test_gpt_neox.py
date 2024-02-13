import unittest

from transformers import GPTNeoXTokenizerFast

from tests._tokenizer_common import TokenizerTesterMixin

import logging


class GPTNeoXTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = GPTNeoXTokenizerFast
    pretrained_name = "EleutherAI/pythia-160m-deduped"

    def setUp(self):
        super().setUp()
