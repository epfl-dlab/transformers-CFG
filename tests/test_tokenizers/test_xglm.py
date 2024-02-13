import unittest

from transformers import XGLMTokenizerFast

from tests._tokenizer_common import TokenizerTesterMixin

import logging


class XGLMTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = XGLMTokenizerFast
    pretrained_name = "facebook/xglm-564M"

    def setUp(self):
        super().setUp()
        self.tokenizer.eos_token_id = None
