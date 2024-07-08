import unittest
from transformers_cfg.cli.cli_main import check_model_support


class TestCliModelSupport(unittest.TestCase):
    def test_supported_model(self):
        model = "gpt2"
        self.assertTrue(check_model_support(model))

    def test_unsupported_model(self):
        model = "bigscience/bloom"
        self.assertFalse(check_model_support(model))


if __name__ == "__main__":
    unittest.main()
