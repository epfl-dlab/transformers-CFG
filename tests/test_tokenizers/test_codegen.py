import unittest

from transformers import CodeGenTokenizerFast

from tests._tokenizer_common import TokenizerTesterMixin

import logging


@unittest.skip("GPTNeoXTokenizerFast is not available for testing")
class CodeGenTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = CodeGenTokenizerFast
    pretrained_name = "microsoft/phi-1_5"

    def setUp(self):
        super().setUp()
