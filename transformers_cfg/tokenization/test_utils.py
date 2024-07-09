from unittest import TestCase
from transformers import AutoTokenizer
from transformers_cfg.tokenization.utils import get_tokenizer_charset


class TestGetTokenizerCharset(TestCase):
    def test_get_tokenizer_charset(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        charset = get_tokenizer_charset(tokenizer)
        expected_charset = set(
            "abcdefghijklmnopqrstuvwxyz0123456789!#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        )
        self.assertEqual(charset, expected_charset)
