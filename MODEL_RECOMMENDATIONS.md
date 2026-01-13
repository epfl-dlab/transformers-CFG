# Model Recommendations for Examples

Different examples require different model capabilities. Here's a guide:

## ✅ Works Great with Llama 3.2 1B (Base)

These examples work well with the smaller 1B model:

### 1. JSON Generation
```bash
python examples/generate_json.py
```
**Model:** `meta-llama/Llama-3.2-1B` ✅
**Why:** Simple structure, common in training data

### 2. Pipeline JSON
```bash
python examples/pipeline_json.py
```
**Model:** `meta-llama/Llama-3.2-1B` ✅
**Why:** Same as above

---

## ⚠️ Better with Larger Models (3B+)

These examples need more model capacity for good results:

### 3. GeoQuery Generation
```bash
# With 1B - produces simple but valid output
python examples/generate_geo_query.py

# Better results with 3B
python examples/generate_geo_query.py --model-id meta-llama/Llama-3.2-3B

# Best results with 7B
python examples/generate_geo_query.py --model-id mistralai/Mistral-7B-v0.1
```

**Why it's harder:**
- Complex semantic parsing
- Domain-specific knowledge (US geography)
- Nested function structures
- Specific entity names

**1B Results:** `answer(0)` - grammatically valid but semantically trivial
**3B+ Results:** `answer(smallest_one(area_1(stateid('colorado'))))` - meaningful queries

### 4. PDDL Generation (Planning)
```bash
# With 1B - may produce simple structures
python examples/generate_pddl.py

# Better with larger model
python examples/generate_pddl.py --model-id meta-llama/Llama-3.2-3B
```

**Why it's harder:**
- Planning domain language
- Action sequences
- Domain-specific vocabulary

### 5. SMILES Generation (Chemistry)
```bash
# With 1B - produces valid but simple molecules
python examples/generate_smiles.py

# Better with larger model
python examples/generate_smiles.py --model-id meta-llama/Llama-3.2-3B
```

**Why it's harder:**
- Chemistry notation
- Valid molecular structures
- Domain-specific constraints

---

## Model Size vs Quality Trade-off

| Task Complexity | Recommended Model | Size | Download | Quality |
|----------------|-------------------|------|----------|---------|
| **Simple** (JSON) | Llama 3.2 1B | 2.5GB | 5-10 min | Good ✅ |
| **Medium** (SMILES) | Llama 3.2 3B | 6GB | 15-20 min | Very Good |
| **Complex** (GeoQuery, PDDL) | Mistral 7B | 14GB | 30-60 min | Excellent |

---

## Why Llama 3.2 1B Struggles with GeoQuery

### The Grammar
GeoQuery has very specific structure:
```
answer(smallest_one(area_1(stateid('colorado'))))
```

### What 1B Produces
```
answer(0)
```

This happens because:
1. **Grammar defines NUM ::= "0"** - the only number allowed
2. **Model takes easiest path** - `answer(0)` is simpler than nested functions
3. **Lacks domain knowledge** - doesn't know US state relationships
4. **Base model limitations** - no instruction following

### What Mistral 7B Produces
```
answer(smallest_one(area_1(stateid('colorado'))))
```

Why it works better:
1. **7x more parameters** - better semantic understanding
2. **Seen more diverse data** - including structured queries
3. **Better at long-range dependencies** - can generate nested structures

---

## Recommendations

### For Development/Testing
**Use Llama 3.2 1B for:**
- Quick iteration on JSON examples
- Testing grammar constraints work correctly
- Verifying refactored code functions

### For Demonstrations
**Use Llama 3.2 3B for:**
- Better quality across all examples
- Still reasonably fast downloads (6GB)
- Good balance of size vs capability

### For Production/Research
**Use Mistral 7B or larger for:**
- Best quality semantic parsing
- Complex domain-specific generation
- When accuracy matters more than speed

---

## How to Switch Models

All examples support `--model-id` flag:

```bash
# Default (1B - fast, basic quality)
python examples/generate_geo_query.py

# Better quality (3B - moderate speed)
python examples/generate_geo_query.py --model-id meta-llama/Llama-3.2-3B

# Best quality (7B - slower)
python examples/generate_geo_query.py --model-id mistralai/Mistral-7B-v0.1

# Any HuggingFace model (use base models, not Instruct)
python examples/generate_geo_query.py --model-id your-model-id
```

**Important:** Always use **base models**, not Instruct variants, for grammar-constrained generation.

---

## Grammar Constraints Still Work!

Even with 1B producing `answer(0)`:
- ✅ Grammar constraint is functioning correctly
- ✅ Invalid tokens are masked out
- ✅ Output is grammatically valid
- ⚠️ Output is just semantically trivial

The grammar system works perfectly - it's the model's knowledge that's limited.

---

## Summary

| Example | 1B Quality | 3B Quality | 7B Quality | Recommendation |
|---------|-----------|-----------|-----------|----------------|
| JSON | Good ✅ | Excellent | Excellent | 1B is fine |
| Pipeline JSON | Good ✅ | Excellent | Excellent | 1B is fine |
| SMILES | Basic ⚠️ | Good | Very Good | 3B+ preferred |
| PDDL | Basic ⚠️ | Good | Very Good | 3B+ preferred |
| GeoQuery | Trivial ❌ | Good | Excellent | 3B+ needed |

**Default to 1B for development, upgrade to 3B+ when you need better semantic quality.**
