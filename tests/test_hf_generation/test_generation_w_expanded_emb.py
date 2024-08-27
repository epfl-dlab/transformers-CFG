from unittest import TestCase
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_cfg.token_grammar_recognizer import IncrementalTokenRecognizer
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

MODEL_IDS = [
    "JackFram/llama-68m",
]


def check_parentheses(generation):
    stack = []
    for char in generation:
        if char == "(":
            stack.append(char)
        elif char == ")":
            if not stack:
                return False
            stack.pop()
    return not stack


class TestGreedyDecoding(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {}
        cls.tokenizers = {}
        for model_id in MODEL_IDS:
            cls.models[model_id] = AutoModelForCausalLM.from_pretrained(model_id)
            cls.tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
            cls.tokenizers[model_id].pad_token = cls.tokenizers[model_id].eos_token
            # we expand the embedding layer to simulate the case where the model has a larger embedding layer than the tokenizer
            cls.models[model_id].resize_token_embeddings(
                10 + len(cls.tokenizers[model_id])
            )

    def test_generate_only_number(self):
        # test greedy decoding with grammar constraints
        grammar_str = """
        root ::= [0-9]+
        """

        for model_id in MODEL_IDS:
            model = self.models[model_id]
            tokenizer = self.tokenizers[model_id]

            grammar = IncrementalTokenRecognizer(
                grammar_str, start_rule_name="root", tokenizer=tokenizer
            )
            grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

            prefix = "This is a valid number:"

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
                top_p=0.92,
                top_k=5,
                logits_processor=[grammar_processor],
                repetition_penalty=100.0,
                early_stopping=True,
            )

            generations = tokenizer.batch_decode(
                output[:, input_ids.shape[1] :], skip_special_tokens=True
            )
            self.assertTrue(
                generations[0].isdigit(), f"generations: {generations} is not a number"
            )

    def test_generate_balanced_parenthesis(self):
        # test greedy decoding with grammar constraints
        grammar_str = """
        root ::= "(" root ")" | ""
        """

        for model_id in MODEL_IDS:
            model = self.models[model_id]
            tokenizer = self.tokenizers[model_id]

            grammar = IncrementalTokenRecognizer(
                grammar_str, start_rule_name="root", tokenizer=tokenizer
            )
            grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

            prefix = "This is a valid json:"

            input_ids = tokenizer(
                [prefix], add_special_tokens=False, return_tensors="pt", padding=True
            )["input_ids"]

            output = model.generate(
                input_ids,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
                max_new_tokens=40,
                top_p=0.92,
                top_k=5,
                logits_processor=[grammar_processor],
                repetition_penalty=100.0,
                early_stopping=True,
            )

            generation: str = tokenizer.batch_decode(
                output[:, input_ids.shape[1] :], skip_special_tokens=True
            )[0]

            self.assertTrue(
                check_parentheses(generation),
                f"generations: {generation} is not a balanced parenthesis",
            )
