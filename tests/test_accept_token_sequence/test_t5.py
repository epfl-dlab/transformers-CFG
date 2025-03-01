from transformers import T5TokenizerFast
from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)

# @unittest.skip("T5Tokenizer's mapping is not well defined, not working")
class TestT5Tokenizer(TokenizerTesterMixin):
    tokenizer_class = T5TokenizerFast
    pretrained_name = "t5-small"

    def setup(self):
        self.setup_tokenizer()


class TestT5TokenizerUnkToken:
    def test_unk_token(self):
        tokenizer = T5TokenizerFast.from_pretrained("t5-small")

        unk_token_id = tokenizer.unk_token_id
        unk_token = tokenizer.unk_token

        # open curly brace is an unk token
        curly_brace_open = "{"
        # we take the 2nd token because the first token is the space token
        curly_brace_open_id = tokenizer.encode(curly_brace_open)[1]
        assert curly_brace_open_id == unk_token_id

        curly_brace_close = "}"
        curly_brace_close_id = tokenizer.encode(curly_brace_close)[1]
        assert curly_brace_close_id == unk_token_id

        eos_token_id = tokenizer.eos_token_id
        # tab in t5 signifies the end of a line
        tab = "\t"
        tab_id = tokenizer.encode(tab)[0]
        assert tab_id == eos_token_id

        # newline in t5 signifies the end of a line
        newline = "\n"
        newline_id = tokenizer.encode(newline)[0]
        assert newline_id == eos_token_id
