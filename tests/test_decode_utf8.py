import unittest
from transformers_cfg.utf8_utils import (
    decode_utf8,
    PartialUTF8,
)  # Make sure to import your function and class


class TestDecodeUTF8(unittest.TestCase):
    def test_decode_single_character(self):
        """Test decoding a single character."""
        utf8_bytes = b"\xe2\x82\xac"  # Euro sign
        expected_code_points = [
            8364,
        ]  # Euro sign code point followed by terminating 0
        result, _ = decode_utf8(utf8_bytes, PartialUTF8())
        self.assertEqual(result, expected_code_points)

    def test_decode_multiple_characters(self):
        """Test decoding a string with multiple UTF-8 characters."""
        utf8_bytes = b"Hello, \xe2\x82\xac!"  # "Hello, €!"
        expected_code_points = [
            72,
            101,
            108,
            108,
            111,
            44,
            32,
            8364,
            33,
        ]
        result, _ = decode_utf8(utf8_bytes, PartialUTF8())
        self.assertEqual(result, expected_code_points)

    def test_handle_incomplete_sequence(self):
        """Test handling of an incomplete UTF-8 sequence."""
        utf8_bytes = b"\xe2"  # Incomplete sequence for the Euro sign
        expected_code_points = []  # Expect a 0 due to incomplete sequence
        result, partial = decode_utf8(utf8_bytes, PartialUTF8())
        self.assertEqual(result, expected_code_points)
        starting_value = int.from_bytes(b"\xe2", "big")  # 226
        offset_value = int.from_bytes(b"\xe0", "big")  # 224
        self.assertEqual(partial.value, starting_value - offset_value)  # 226-224=2
        self.assertEqual(
            partial.n_remain, 2
        )  # Expect n_remain to be 2 because 2 more bytes are needed

    def test_continue_incomplete_sequence(self):
        """Test continuation of decoding with a previously incomplete sequence."""
        utf8_bytes = b"\x82\xac"  # Continuation of the Euro sign
        partial_start = PartialUTF8(
            value=2, n_remain=2
        )  # Simulate a previous state expecting 2 more bytes
        expected_code_points = [
            8364,
        ]  # Completed Euro sign code point
        result, _ = decode_utf8(utf8_bytes, partial_start)
        self.assertEqual(result, expected_code_points)

    def test_empty_string(self):
        """Test handling of an empty string."""
        utf8_bytes = b""
        expected_code_points = []
        result, _ = decode_utf8(utf8_bytes, PartialUTF8())
        self.assertEqual(result, expected_code_points)


if __name__ == "__main__":
    unittest.main()
