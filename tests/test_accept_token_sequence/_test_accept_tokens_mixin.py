import warnings
import pytest
from transformers import PreTrainedTokenizer
from transformers_cfg.token_grammar_recognizer import IncrementalTokenRecognizer
from transformers_cfg.utils import pprint_token_ids


class TokenizerTesterMixin:
    tokenizer_class = None
    pretrained_name = None
    rust_tokenizer_class = None
    test_slow_tokenizer = True
    test_rust_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    from_pretrained_filter = None
    from_pretrained_vocab_key = "vocab_file"
    test_seq2seq = True

    # set to True to test a sentencepiece tokenizer
    test_sentencepiece = False

    # set to True to ignore casing when testing a sentencepiece tokenizer
    # test_sentencepiece must also be set to True
    test_sentencepiece_ignore_case = False

    @pytest.fixture(autouse=True)
    def setup_tokenizer(self):
        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self, **kwargs) -> PreTrainedTokenizer:
        use_fast = kwargs.pop("use_fast", True)
        return self.tokenizer_class.from_pretrained(
            self.pretrained_name, use_fast=use_fast, **kwargs
        )

    def test_json_parsable(self):
        # Test that we can load a JSON object
        with open("examples/grammars/json.ebnf", "r") as file:
            input_text = file.read()
        JsontokenRecognizer = IncrementalTokenRecognizer(
            grammar_str=input_text, start_rule_name="root", tokenizer=self.tokenizer
        )

        valid_json = '{"foo": "bar", "baz": "bat"}'
        token_ids = self.tokenizer.encode(valid_json)
        pprint_token_ids(self.tokenizer, token_ids)

        # check if there is unk token
        for token_id in token_ids:
            if token_id == self.tokenizer.unk_token_id:
                warnings.warn(
                    f"unk token found in input_token_ids: {token_ids}, skipping test"
                )
                return

        acc_state = JsontokenRecognizer._update_state_with_single_token_seq(
            token_ids, as_string=False
        )
        # the json object is complete, so the stacks should be empty
        assert acc_state.stacks == set() or acc_state.stacks == {
            tuple()
        }, f"stacks: {acc_state.stacks}, not empty"

    def test_balanced_parentheses(self):
        # Test that we can recognize a balanced parentheses
        with open("examples/grammars/balanced_parentheses.ebnf", "r") as file:
            input_text = file.read()
        recognizer = IncrementalTokenRecognizer(
            grammar_str=input_text, start_rule_name="root", tokenizer=self.tokenizer
        )

        balanced_parentheses = "((((((((()))))))))"
        token_ids = self.tokenizer.encode(balanced_parentheses)
        pprint_token_ids(self.tokenizer, token_ids)

        # check if there is unk token
        for token_id in token_ids:
            if token_id == self.tokenizer.unk_token_id:
                warnings.warn(
                    f"unk token found in input_token_ids: {token_ids}, skipping test"
                )
                return
        parsing_state = recognizer._update_state_with_single_token_seq(
            token_ids, as_string=False
        )
        # the json object is complete, so the stacks should be empty
        assert parsing_state.stacks == set() or parsing_state.stacks == {
            tuple()
        }, f"stacks: {parsing_state.stacks}, not empty"

    def test_forcing_sequence(self):

        string_to_force = "12345 678 90"

        grammar_str = f"""
        root ::= "{string_to_force}"

        """

        tokenRecognizer = IncrementalTokenRecognizer(
            grammar_str=grammar_str, start_rule_name="root", tokenizer=self.tokenizer
        )

        token_ids = self.tokenizer.encode(string_to_force)
        pprint_token_ids(self.tokenizer, token_ids)

        # check if there is unk token
        for token_id in token_ids:
            if token_id == self.tokenizer.unk_token_id:
                warnings.warn(
                    f"unk token found in input_token_ids: {token_ids}, skipping test"
                )
                return

        acc_state = tokenRecognizer._update_state_with_single_token_seq(
            token_ids, as_string=False
        )
        # the json object is complete, so the stacks should be empty
        assert acc_state.stacks == set() or acc_state.stacks == {
            tuple()
        }, f"stacks: {acc_state.stacks}, not empty"

    def test_emoji(self):
        """
        Test that we can accept emoji
        """

        with open("examples/grammars/emoji.ebnf", "r") as file:
            input_text = file.read()

        tokenRecognizer = IncrementalTokenRecognizer(
            grammar_str=input_text, start_rule_name="root", tokenizer=self.tokenizer
        )

        emoji = "😀😄😂"
        token_ids = self.tokenizer.encode(emoji)
        pprint_token_ids(self.tokenizer, token_ids)

        # check if there is unk token
        for token_id in token_ids:
            if token_id == self.tokenizer.unk_token_id:
                warnings.warn(
                    f"unk token found in input_token_ids: {token_ids}, skipping test"
                )
                return

        accpetance = tokenRecognizer.accept_token_ids(token_ids, as_string=False)

        assert accpetance, f"emoji: {emoji} not accepted, but it should be"
