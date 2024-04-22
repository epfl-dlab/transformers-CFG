from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers_cfg.generation import GrammarConstrainedLogitsProcessor
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.metrics import ConstrainedDecodingMetric
from transformers_cfg.metrics.metrics import ConstrainedDecodingMetricOutput

if __name__ == "__main__":
    metric = ConstrainedDecodingMetric()

    model_id = "gpt2"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Load json grammar
    with open("examples/grammars/json.ebnf", "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Generate
    input_ids = tokenizer(
        [
            "This is a valid json string for http request:",
            "This is a valid json string for shopping cart:",
        ],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )["input_ids"]
    output = model.generate(
        input_ids,
        max_length=30,
        logits_processor=[grammar_processor],
        repetition_penalty=1,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )

    # decode output
    generations = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
    print(generations)

    result: ConstrainedDecodingMetricOutput = metric.compute_from_model_output(output)
    print("Original token probabilities:")
    print(result.df["original_token_probs"].head())

    print("Renormalised token probabilities:")
    print(result.df["renormalised_token_probs"].head())

    print("Total rejection probability gain:")
    print(result.df["total_rejection_prob_gain"].head())

    """
    Original token probabilities:
             Batch 1   Batch 2
    Step 1  0.002313  0.002031
    Step 2  0.250744  0.165687
    Step 3  0.104876  0.090784
    Step 4  0.419352  0.691826
    Step 5  0.203741  0.875446
    Renormalised token probabilities:
             Batch 1   Batch 2
    Step 1  0.620275  0.690802
    Step 2  0.762278  0.493411
    Step 3  0.105184  0.091143
    Step 4  0.428187  0.741440
    Step 5  0.205163  0.902042
    Total rejection probability gain:
             Batch 1   Batch 2
    Step 1  0.996271  0.997059
    Step 2  0.671060  0.664201
    Step 3  0.002922  0.003944
    Step 4  0.020609  0.066890
    Step 5  0.006915  0.029484

    """
