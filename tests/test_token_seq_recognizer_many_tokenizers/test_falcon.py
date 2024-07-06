import unittest

from transformers import AutoTokenizer

from tests._test_token_seq_recognizer_many_tokenizer_common import TokenizerTesterMixin


@unittest.skip("Falcom is not supported and will be removed")
class FalconTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = AutoTokenizer
    pretrained_name = "tiiuae/falcon-7b"

    def setUp(self):
        super().setUp()
