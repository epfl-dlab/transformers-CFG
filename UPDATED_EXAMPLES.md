# Updated Examples - Now Using Llama 3.2 1B

All generation examples have been updated to use **Llama 3.2 1B** (base model) instead of larger models.

## ⚠️ Important: Base Model vs Instruct Model

**We use the BASE model (`meta-llama/Llama-3.2-1B`), NOT the Instruct variant.**

The Instruct model (`meta-llama/Llama-3.2-1B-Instruct`) adds special chat template tokens that conflict with grammar constraints, causing errors like:
```
ValueError: All stacks are empty, so the only token accepted is EOS(128009), but got 128001
```

**Solution:** Always use the base model for grammar-constrained generation:
```python
✅ CORRECT: model_id = "meta-llama/Llama-3.2-1B"
❌ WRONG:   model_id = "meta-llama/Llama-3.2-1B-Instruct"
```

## Why Llama 3.2 1B?

✅ **Much smaller:** ~2.5GB vs ~14GB (Mistral-7B)
✅ **Faster downloads:** 5-10 minutes vs 30-60 minutes
✅ **Faster inference:** Better for testing and demonstrations
✅ **Still capable:** Decent quality for grammar-constrained generation
✅ **Less memory:** Runs on modest GPUs or even CPU

### ⚠️ Model Capability Limitations

**Llama 3.2 1B works great for:**
- ✅ JSON generation (simple structures)
- ✅ Testing grammar constraints
- ✅ Development and quick iteration

**Llama 3.2 1B is limited for:**
- ⚠️ GeoQuery (complex semantic parsing)
- ⚠️ PDDL (planning domain language)
- ⚠️ SMILES (chemistry notation)

For these complex tasks, the 1B model produces grammatically valid but semantically trivial outputs (e.g., `answer(0)` instead of `answer(smallest_one(area_1(stateid('colorado'))))`).

**Recommendation:** Use Llama 3.2 3B or Mistral 7B for complex semantic parsing tasks.

See [MODEL_RECOMMENDATIONS.md](MODEL_RECOMMENDATIONS.md) for detailed guidance on which model to use for each example.

## Updated Examples

All these examples now default to `meta-llama/Llama-3.2-1B` (base model):

### 1. JSON Generation
```bash
python examples/generate_json.py
```
**Before:** Mistral-7B (~14GB)
**After:** Llama 3.2 1B (~2.5GB)

---

### 2. Pipeline JSON Generation
```bash
python examples/pipeline_json.py
```
**Before:** Mistral-7B (~14GB)
**After:** Llama 3.2 1B (~2.5GB)

---

### 3. SMILES Generation (Chemistry)
```bash
python examples/generate_smiles.py
python examples/generate_smiles.py --smiles-type isocyanates
```
**Before:** Mistral-7B (~14GB)
**After:** Llama 3.2 1B (~2.5GB)

---

### 4. PDDL Generation (Planning)
```bash
python examples/generate_pddl.py
python examples/generate_pddl.py --pddl-type blocks
```
**Before:** Mistral-7B (~14GB)
**After:** Llama 3.2 1B (~2.5GB)

---

### 5. GeoQuery Generation
```bash
python examples/generate_geo_query.py
```
**Before:** Mistral-7B (~14GB)
**After:** Llama 3.2 1B (~2.5GB)

---

### 6. Demo JSON Generation
```bash
python demo_json_generation.py
```
**Before:** GPT2 (~500MB)
**After:** Llama 3.2 1B (~2.5GB)

## How to Run

### First Time Setup
The model will download automatically on first run (~2.5GB):

```bash
python examples/generate_json.py
```

Expected output:
```
Using device: cuda
✓ Loading Llama 3.2 1B model and tokenizer...
Downloading model... (first time only)
✓ Generating JSON...
{'name': 'John', 'age': 30}
```

### Subsequent Runs
Much faster (no download needed):

```bash
python examples/generate_json.py
```

## Override Model (Optional)

You can still use other models if you prefer:

```bash
# Use GPT2 (smallest)
python examples/generate_json.py --model-id gpt2

# Use Mistral-7B (if you have it downloaded)
python examples/generate_json.py --model-id mistralai/Mistral-7B-v0.1

# Use any HuggingFace model (base models work best)
python examples/pipeline_json.py --model-id meta-llama/Llama-3.2-3B
```

## System Requirements

### Minimum (CPU only):
- RAM: 4GB
- Disk: 5GB free
- Time: 30-60 seconds per generation

### Recommended (with GPU):
- GPU: 4GB VRAM (e.g., GTX 1660, RTX 3050)
- RAM: 8GB
- Disk: 5GB free
- Time: 2-5 seconds per generation

### Optimal (with GPU):
- GPU: 8GB+ VRAM (e.g., RTX 3070, RTX 4060)
- RAM: 16GB
- Disk: 10GB free
- Time: 1-2 seconds per generation

## Comparison Table

| Model | Size | Download Time | Inference Speed | Quality |
|-------|------|---------------|-----------------|---------|
| **Llama 3.2 1B** (new default) | 2.5GB | 5-10 min | Fast | Good |
| GPT2 | 500MB | 2-3 min | Very Fast | Basic |
| Mistral-7B (old default) | 14GB | 30-60 min | Moderate | Excellent |
| Llama 3.2 3B | 6GB | 15-20 min | Moderate | Very Good |

## Migration Notes

### If You Have Old Examples
The old examples with Mistral-7B will still work, but we recommend updating to Llama 3.2 1B for:
- Faster development/testing
- Lower memory requirements
- Easier sharing and collaboration

### No Code Changes Needed
All the refactored code works identically with any model. Only the model ID changed:

```python
# Old
model_id = "mistralai/Mistral-7B-v0.1"

# New (base model, not Instruct)
model_id = "meta-llama/Llama-3.2-1B"
```

**Critical:** Use base models for grammar-constrained generation. Instruct models have special tokens that interfere with grammars.

## Testing

Run a quick test:

```bash
# Quick validation test (no model download)
python examples/accept.py -g examples/grammars/json.ebnf -s '{"test": 123}' -m sentence

# Full generation test (downloads Llama 3.2 1B first time)
python examples/generate_json.py
```

## Troubleshooting

### Model Download Fails
```bash
# Try with explicit cache directory
export HF_HOME=/path/to/cache
python examples/generate_json.py
```

### Out of Memory
```bash
# Use CPU instead of GPU
CUDA_VISIBLE_DEVICES="" python examples/generate_json.py
```

### Slow Generation
- First time: Model is downloading (normal)
- Subsequent runs: Try using GPU if available
- Consider using smaller max_new_tokens

### Grammar Constraint Errors
If you see errors like "All stacks are empty, so the only token accepted is EOS":
```bash
# Make sure you're using the BASE model, not Instruct
# ❌ Wrong: meta-llama/Llama-3.2-1B-Instruct
# ✅ Correct: meta-llama/Llama-3.2-1B
```

Instruct models add special chat tokens that conflict with grammar constraints.

## Summary

✅ All examples updated to use Llama 3.2 1B
✅ 5-6x smaller downloads (2.5GB vs 14GB)
✅ Faster testing and development
✅ All refactored code works perfectly
✅ Can still use any model with `--model-id` flag
