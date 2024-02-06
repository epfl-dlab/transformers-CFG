import unittest
import warnings

from transformers import PreTrainedTokenizer
from transformers_cfg.token_grammar_recognizer import IncrementalTokenGrammarRecognizer

from transformers_cfg.recognizer import GrammarRecognizer

from transformers_cfg.parser import parse_ebnf
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

    def setUp(self):
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
        JsontokenRecognizer = IncrementalTokenGrammarRecognizer(
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

        stacks = JsontokenRecognizer._consume_token_ids(
            token_ids, JsontokenRecognizer.grammar.stacks, as_string=False
        )
        # the json object is complete, so the stacks should be empty
        self.assertTrue(stacks == [] or stacks == [[]], f"stacks: {stacks}, not empty")

    # def test_beam_search_low_memory(self):
    #     # Check that choosing 'low_memory' does not change the model output
    #     for model_class in self.all_generative_model_classes:
    #         if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
    #             self.skipTest("Won't fix: old model with different cache format")
    #         if any(
    #             model_name in model_class.__name__.lower()
    #             for model_name in [
    #                 "bloom",
    #                 "ctrl",
    #                 "gptbigcode",
    #                 "transo_xl",
    #                 "xlnet",
    #                 "cpm",
    #             ]
    #         ):
    #             self.skipTest("May fix in the future: need model-specific fixes")
    #         config, input_ids, attention_mask, max_length = self._get_input_ids_and_config(batch_size=2)
    #         # batch_size=1 is ok, but batch_size>1 will cause non-identical output
    #
    #         config.use_cache = True
    #         config.is_decoder = True
    #
    #         # test output equality of low versus high memory
    #         model = model_class(config).to(torch_device).eval()
    #
    #         low_output = model.generate(input_ids, max_new_tokens=8, num_beams=5, early_stopping=True, low_memory=True)
    #
    #         high_output = model.generate(
    #             input_ids, max_new_tokens=8, num_beams=5, early_stopping=True, low_memory=False
    #         )
    #         self.assertListEqual(low_output.tolist(), high_output.tolist())
