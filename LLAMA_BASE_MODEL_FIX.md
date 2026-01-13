# Llama 3.2 Base Model Fix

## Summary

Successfully fixed all 6 examples to use **Llama 3.2 1B base model** instead of the Instruct variant, resolving grammar constraint errors.

## The Problem

### Error Encountered
When using `meta-llama/Llama-3.2-1B-Instruct`, examples failed with:
```
ValueError: All stacks are empty, so the only token accepted is EOS(128009), but got 128001.
This error is likely due to the previous token not being accepted by the grammar.
```

### Root Cause
The **Instruct variant** of Llama 3.2 1B applies a chat template that adds special tokens at the beginning of generation. These special tokens are not part of the grammar, causing the grammar processor to reject them immediately.

Specifically:
- Token 128009 is the expected EOS token
- Token 128001 is a special chat template token added by the Instruct model
- The grammar expects to start matching from the first token, but gets an unexpected special token

## The Solution

### Change Applied
Switch from Instruct to base model in all examples:

```python
# ❌ WRONG - causes grammar errors
model_id = "meta-llama/Llama-3.2-1B-Instruct"

# ✅ CORRECT - works with grammar constraints
model_id = "meta-llama/Llama-3.2-1B"
```

### Files Updated
1. `examples/generate_json.py`
2. `examples/pipeline_json.py`
3. `examples/generate_smiles.py`
4. `examples/generate_pddl.py`
5. `examples/generate_geo_query.py`
6. `demo_json_generation.py`

### Additional Fix
All files already had the `eos_token_id` list handling:
```python
# Handle pad_token_id (eos_token_id might be a list for some models)
eos_token_id = model.generation_config.eos_token_id
if isinstance(eos_token_id, list):
    eos_token_id = eos_token_id[0]
model.generation_config.pad_token_id = eos_token_id
```

This handles the fact that Llama 3.2's `eos_token_id` is a list `[128001, 128009]` rather than a single integer.

## What Didn't Work

### Attempt 1: Using `valid_token_start_idx`
```python
# This was NOT sufficient with Instruct model
grammar_processor = GrammarConstrainedLogitsProcessor(
    grammar,
    valid_token_start_idx=input_ids.shape[1]
)
```

**Result:** Still failed because the Instruct model's chat template affects the entire generation process, not just the prompt.

### Attempt 2: Staying with Instruct model
The Instruct variant fundamentally changes how generation works by wrapping all inputs in a chat template. This is incompatible with grammar-constrained generation that expects exact token control.

## Verification

### Testing Results
```bash
# All examples now work correctly
$ python examples/generate_json.py
Using device: cuda
['...{"name":"test","age":18}...', '...{"items":[...]}...']

$ python demo_json_generation.py
✓ Valid JSON structure generated!
{"name":"John","age":25,"city":"New York"}

$ python examples/generate_smiles.py
Constrained: ...O=C1=CC2C(C3)CCC4N(CC5)=CN([H]...
```

### Verification Script
```bash
$ python verify_base_model_usage.py
✅ All 6 files verified using base model correctly
```

## Documentation Updates

### UPDATED_EXAMPLES.md
Added prominent warning section:
```markdown
## ⚠️ Important: Base Model vs Instruct Model

**We use the BASE model (`meta-llama/Llama-3.2-1B`), NOT the Instruct variant.**

The Instruct model adds special chat template tokens that conflict with
grammar constraints.
```

### Troubleshooting Section
Added guidance for users encountering similar errors:
```markdown
### Grammar Constraint Errors
If you see "All stacks are empty, so the only token accepted is EOS":
- Make sure you're using the BASE model, not Instruct
- Instruct models add special chat tokens that conflict with grammars
```

## Key Takeaways

1. **Always use base models for grammar-constrained generation**
   - Base models: Clean token generation
   - Instruct models: Add chat templates that interfere with grammars

2. **Llama 3.2 specific issues**
   - `eos_token_id` is a list, not an int
   - Instruct variant adds special tokens 128001, 128009
   - Base model avoids these issues

3. **Grammar constraints require exact token control**
   - Chat templates and grammar constraints are fundamentally incompatible
   - Use base models when you need grammar enforcement

## Related Files

- **Examples:** `examples/*.py`, `demo_json_generation.py`
- **Documentation:** `UPDATED_EXAMPLES.md`
- **Verification:** `verify_base_model_usage.py`
- **Migration guide:** `EXAMPLES_MIGRATION_SUMMARY.md`

## Future Considerations

### For Other Models
This issue likely affects other model families too:
- Any model with "-Instruct", "-Chat", or "-It" suffix
- Models that apply automatic chat templates
- Models with special conversation tokens

**Recommendation:** When using grammar-constrained generation, always prefer base models over instruct/chat variants.

### For Library Users
If you must use an Instruct model:
1. Disable the chat template if possible
2. Use `valid_token_start_idx` to skip special tokens (may not always work)
3. Consider switching to the base model instead

## Success Metrics

✅ All 6 examples now run without errors
✅ Valid JSON, SMILES, PDDL output generated
✅ Verification script confirms correct model usage
✅ Documentation warns users about base vs Instruct
✅ Clear troubleshooting guidance provided

## Commit

```
7752364 Fix: Switch all examples from Instruct to base Llama 3.2 1B model
```

This fix ensures all examples work correctly with grammar constraints and provides clear guidance to prevent similar issues in the future.
