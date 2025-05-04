from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

# Track variable states: unallocated, allocated, freed
def get_memory_state(input_str):
    """
    Rudimentary memory state tracker from generated string.
    Assumes a single pointer variable named 'ptr'.
    """
    allocated = "malloc" in input_str or "calloc" in input_str
    freed = "free(ptr" in input_str
    used = "ptr" in input_str and not ("malloc" in input_str or "free(ptr" in input_str)

    if freed:
        return "freed"
    elif allocated:
        return "allocated"
    elif used:
        return "used_without_alloc"
    else:
        return "unallocated"

def prefix_allowed_tokens_fn(batch_id, input_ids):
    input_str = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    state = get_memory_state(input_str)

    tok = tokenizer
    allow_tokens = []

    malloc_token = tok("malloc", add_special_tokens=False).input_ids[0]
    free_token = tok("free", add_special_tokens=False).input_ids[0]
    ptr_token = tok("ptr", add_special_tokens=False).input_ids[0]

    if state == "unallocated":
        # Only malloc is allowed
        allow_tokens = list(range(tok.vocab_size))
    elif state == "allocated":
        # Allow free or use of ptr
        allow_tokens = list(range(tok.vocab_size))
        allow_tokens.remove(malloc_token)
    elif state == "freed":
        # Don't allow ptr usage or free again
        allow_tokens = [
            i for i in range(tok.vocab_size) if i != ptr_token and i != free_token
        ]
    elif state == "used_without_alloc":
        # Penalize generation since ptr is used before malloc
        allow_tokens = [malloc_token]  # optionally reset
    else:
        # Fallback: allow everything
        allow_tokens = list(range(tok.vocab_size))

    return allow_tokens

prompt = "Write a C function to unsafely allocate and free memory using ptr. Code:\n"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=5,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
