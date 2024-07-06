import unittest

from transformers import GPT2TokenizerFast

from tests._test_token_seq_recognizer_many_tokenizer_common import TokenizerTesterMixin


class GPT2TokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = GPT2TokenizerFast
    pretrained_name = "gpt2"

    def setUp(self):
        super().setUp()
