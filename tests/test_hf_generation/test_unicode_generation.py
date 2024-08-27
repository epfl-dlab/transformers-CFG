from unittest import TestCase
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.token_grammar_recognizer import IncrementalTokenRecognizer
from transformers_cfg.token_grammar_recognizer import AbsTokenRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor


UNICODE_MODEL_IDS = [
    "JackFram/llama-68m",
]


class TestDetectUnicode(TestCase):
    def test_detect_unicode(self):
        # Test with a string containing only ASCII characters
        self.assertFalse(AbsTokenRecognizer.detect_unicode("Hello, world!"))

        # Test with a string containing Unicode characters
        self.assertTrue(AbsTokenRecognizer.detect_unicode("你好，世界！"))

        # Test with an empty string
        self.assertFalse(AbsTokenRecognizer.detect_unicode(""))

        # Test with a string containing a mix of ASCII and Unicode characters
        self.assertTrue(AbsTokenRecognizer.detect_unicode("Hello, René!"))


class TestGreedyDecoding(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {}
        cls.tokenizers = {}
        for model_id in UNICODE_MODEL_IDS:
            cls.models[model_id] = AutoModelForCausalLM.from_pretrained(model_id)
            cls.tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
            cls.tokenizers[model_id].pad_token = cls.tokenizers[model_id].eos_token

    def test_generate_emoji(self):
        # test greedy decoding with grammar constraints
        grammar_str = """
        root ::= "🤣"
        """

        for model_id in UNICODE_MODEL_IDS:
            model = self.models[model_id]
            tokenizer = self.tokenizers[model_id]

            grammar = IncrementalTokenRecognizer(
                grammar_str, start_rule_name="root", tokenizer=tokenizer
            )
            grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

            prefix = "Generate an emoji:"

            input_ids = tokenizer(
                [prefix], add_special_tokens=False, return_tensors="pt", padding=True
            )["input_ids"]

            output = model.generate(
                input_ids,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
                max_new_tokens=40,
                logits_processor=[grammar_processor],
                early_stopping=True,
            )

            generations = tokenizer.batch_decode(
                output[:, input_ids.shape[1] :], skip_special_tokens=True
            )
            self.assertTrue(
                generations[0] == "🤣", f"generations: {generations} is not equal to '🤣'"
            )
