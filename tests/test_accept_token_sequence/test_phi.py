from transformers import CodeGenTokenizerFast
from tests.test_accept_token_sequence._test_accept_tokens_mixin import (
    TokenizerTesterMixin,
)

# @unittest.skip("CodeGen is not supported and will be removed")
class TestPhiTokenizer(TokenizerTesterMixin):
    tokenizer_class = CodeGenTokenizerFast
    pretrained_name = "Transformers-CFG/phi-1_5-tokenizer"

    def setup(self):
        self.setup_tokenizer()
