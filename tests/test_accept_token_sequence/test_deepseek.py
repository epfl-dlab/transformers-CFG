import unittest

from transformers import AutoTokenizer

from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)


# @unittest.skip("CodeGen is not supported and will be removed")
class DeepSeekTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = AutoTokenizer
    pretrained_name = "deepseek-ai/deepseek-coder-1.3b-base"

    def setUp(self):
        super().setUp()
