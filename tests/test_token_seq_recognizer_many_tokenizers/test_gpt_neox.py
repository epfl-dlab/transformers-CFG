import unittest

from transformers import GPTNeoXTokenizerFast

from tests._test_token_seq_recognizer_many_tokenizer_common import TokenizerTesterMixin

import logging


@unittest.skip("GPTNeoX is not supported and will be removed")
class GPTNeoXTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = GPTNeoXTokenizerFast
    pretrained_name = "EleutherAI/pythia-160m-deduped"

    def setUp(self):
        super().setUp()
