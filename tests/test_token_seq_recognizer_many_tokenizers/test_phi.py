import unittest

from transformers import CodeGenTokenizerFast

from tests._test_token_seq_recognizer_many_tokenizer_common import TokenizerTesterMixin


# @unittest.skip("CodeGen is not supported and will be removed")
class PhiTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = CodeGenTokenizerFast
    pretrained_name = "microsoft/phi-1_5"

    def setUp(self):
        super().setUp()
