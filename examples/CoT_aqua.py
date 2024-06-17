import re
import torch
import argparse
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate calflow strings")
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        help="Model ID",
    )
    parser.add_argument("--device", type=str, help="Device to put the model on")
    return parser.parse_args()


def create_prompts(sample):
    cot_in_context = "Think step-by-step, Question: How many keystrokes are needed to type the numbers from 1 to 500?\nAnswer Choices: A)1156 B)1392 C)1480 D)1562 E)1788\nReasoning: There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90 * 2 + 401 * 3 = 1392.\nAnswer: B);\n"
    in_context = "Question: How many keystrokes are needed to type the numbers from 1 to 500?\nAnswer Choices: A)1156 B)1392 C)1480 D)1562 E)1788.\nAnswer: B);\n"

    sample_text = f"Question: {sample['question']}\nAnswer Choices: {' '.join(sample['options'])}\n"

    prompt_cot = f"{cot_in_context}{sample_text}Reasoning: "
    sample["prompt_cot"] = prompt_cot

    prompt_1_shot = f"{in_context}{sample_text}Answer: "
    sample["prompt_1_shot"] = prompt_1_shot

    return sample


def extract_answers(batch, generations, answers):
    def _parse_prediction(prediction):
        pattern = r"[A-E]\)"
        predcted_answer = re.search(pattern, prediction)
        return predcted_answer[0][0] if predcted_answer else ""

    batch_size = len(batch["prompt_cot"])

    for i in range(batch_size):
        prompt_1_shot = batch["prompt_1_shot"][i]
        prompt_cot = batch["prompt_cot"][i]
        batch_size = len(batch["prompt_cot"])

        unconstrained_prediction = generations[i][len(prompt_cot) :]
        constrained_cot_prediction = generations[i + batch_size][len(prompt_cot) :]
        constrained_mcqa_prediction = generations[i + 2 * batch_size][
            len(prompt_1_shot) :
        ]

        answers["gt"].append(batch["correct"][i])
        answers["unconstrained"].append(_parse_prediction(unconstrained_prediction))
        answers["constrained_cot"].append(_parse_prediction(constrained_cot_prediction))
        answers["constrained_mcqa"].append(
            _parse_prediction(constrained_mcqa_prediction)
        )


def count_empty(predictions):
    return sum(1 for pred in predictions if not pred)


def load_grammar_processor(grammar_path, tokenizer):
    with open(grammar_path, "r") as file:
        grammar_str = file.read()

    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
    return grammar_processor


def main():
    args = parse_args()
    model_id = args.model_id

    # Detect if GPU is available, otherwise use CPU
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Load model to defined device
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    test_dataset = load_dataset("deepmind/aqua_rat", split="test")
    test_dataset = test_dataset.map(create_prompts)

    max_new_tokens = 300
    batch_size = 8

    answers = defaultdict(list)

    for i, batch in enumerate(tqdm(test_dataset.iter(batch_size=batch_size))):
        # Load grammars
        cot_grammar_processor = load_grammar_processor(
            "examples/grammars/chain_of_thought_mcqa.ebnf", tokenizer
        )
        mcqa_grammar_processor = load_grammar_processor(
            "examples/grammars/mcqa.ebnf", tokenizer
        )

        input_ids_1_shot = tokenizer(
            batch["prompt_1_shot"],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )["input_ids"].to(device)

        input_ids_cot = tokenizer(
            batch["prompt_cot"],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )["input_ids"].to(device)

        unconstrained_output = model.generate(
            input_ids_cot,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            num_return_sequences=1,
        )

        constrained_output_cot = model.generate(
            input_ids_cot,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            logits_processor=[cot_grammar_processor],
            repetition_penalty=1.1,
            num_return_sequences=1,
        )

        constrained_output_mcqa = model.generate(
            input_ids_1_shot,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            logits_processor=[mcqa_grammar_processor],
            repetition_penalty=1.1,
            num_return_sequences=1,
        )

        # decode outputs (possibly of different lengths across decoding modes)
        generations = (
            tokenizer.batch_decode(unconstrained_output, skip_special_tokens=True)
            + tokenizer.batch_decode(constrained_output_cot, skip_special_tokens=True)
            + tokenizer.batch_decode(constrained_output_mcqa, skip_special_tokens=True)
        )

        extract_answers(batch, generations, answers)

    print(
        f"Unconstrained accuracy: {accuracy_score(y_true=answers['gt'], y_pred=answers['unconstrained']):.3f}, empty: {count_empty(answers['unconstrained'])} out of {len(answers['unconstrained'])}",
    )
    print(
        f"Constrained accuracy (COT): {accuracy_score(y_true=answers['gt'], y_pred=answers['constrained_cot']):.3f}, empty: {count_empty(answers['constrained_cot'])} out of {len(answers['constrained_cot'])}"
    )
    print(
        f"Constrained accuracy (MCQA): {accuracy_score(y_true=answers['gt'], y_pred=answers['constrained_mcqa']):.3f}, , empty: {count_empty(answers['constrained_mcqa'])} out of {len(answers['constrained_mcqa'])}"
    )


if __name__ == "__main__":
    main()
