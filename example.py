import torch
from datasets import load_dataset
from transformers import AutoTokenizer, WhisperForConditionalGeneration, AutoProcessor
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

if __name__ == "__main__":
    # Set device: use GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model identifier
    model_id = "openai/whisper-tiny.en"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Define grammar string
    grammar_str = """
    root   ::= "The animal is a " animal "."
    animal ::= "cat" | "fish"
    """

    # Create grammar constraint and logits processor
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

    # Define prompts
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )

    inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # Generate constrained text
    generated_ids = model.generate(
        input_features,
        max_length=50,
        logits_processor=[grammar_processor],
    )

    # Decode and print generated text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(transcription)

# The animal is a cat.
