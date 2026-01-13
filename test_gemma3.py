import torch
import transformers
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,  # Same goes for the auto class
    GenerationConfig,
    # BitsAndBytesConfig
)
import transformers_cfg
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
import gc
import platform
import traceback

MODEL_ID = "google/gemma-3-1b-it"
SIMPLE_GRAMMAR = 'root ::= ("A" | "B")'  # Simple grammar
START_RULE = "root"

# Quantization
USE_4BIT = False
USE_8BIT = False

# Environment Info
print("-" * 60)
print("ENVIRONMENT:")
print(f"- Python:       {platform.python_version()}")
print(f"- transformers: {transformers.__version__}")
print(f"- torch:        {torch.__version__}")
try:
    import accelerate

    print(f"- accelerate:   {accelerate.__version__}")
except ImportError:
    print("- accelerate:   Not Installed")
try:
    import bitsandbytes

    print(f"- bitsandbytes: {bitsandbytes.__version__}")
except ImportError:
    print("- bitsandbytes: Not Installed (Needed for quantization)")
try:
    print(f"- trans-cfg:    {transformers_cfg.__version__}")
except AttributeError:
    print("- trans-cfg:    Installed (version attribute missing)")
print("-" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Test
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise AttributeError("Tokenizer missing chat template")

    print("Loading model (Gemma3ForCausalLM)...")
    model = Gemma3ForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        quantization_config=None,  # No quantization
        trust_remote_code=True,
    ).eval()

    model_vocab = getattr(model.config, "vocab_size", "N/A")
    tokenizer_vocab = getattr(tokenizer, "vocab_size", "N/A")
    print(
        f"(Post-load check) Model vocab: {model_vocab}, Tokenizer vocab: {tokenizer_vocab}"
    )
    if model_vocab != tokenizer_vocab:
        print("Post-load vocab size mismatch detected!")

    # Grammar Processor
    print("Creating grammar processor...")
    grammar_constraint = IncrementalGrammarConstraint(
        SIMPLE_GRAMMAR, START_RULE, tokenizer
    )
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar_constraint)
    logits_processor_list = [grammar_processor]

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Output A or B."}]}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(model.device)
    input_ids_len = input_ids.shape[1]

    gen_config = GenerationConfig(
        max_new_tokens=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    # Expected to Fail
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            generation_config=gen_config,
            logits_processor=logits_processor_list,
        )

    output_token_ids = outputs[:, input_ids_len:]
    output_text = tokenizer.decode(output_token_ids[0], skip_special_tokens=True)
    print(f"Output token IDs: {output_token_ids}")
    print(f"Output text: {output_text}")

except ValueError as e:
    print(f"ERROR Type: {type(e).__name__}")
    print(f"ERROR Message: {e}")
    traceback.print_exc()
    print("-" * 35 + "\n")
except Exception as e:
    print(f"ERROR Type: {type(e).__name__}")
    print(f"ERROR Message: {e}")
    traceback.print_exc()
    print("-" * 35 + "\n")
finally:
    del model
    del tokenizer
    del grammar_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
