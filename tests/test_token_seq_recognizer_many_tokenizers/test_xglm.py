import unittest

from transformers import XGLMTokenizerFast

from tests._test_token_seq_recognizer_many_tokenizer_common import TokenizerTesterMixin

import logging


@unittest.skip("Not Supported and Will be removed")
class XGLMTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = XGLMTokenizerFast
    pretrained_name = "facebook/xglm-564M"

    def setUp(self):
        super().setUp()
        self.tokenizer.eos_token_id = None
