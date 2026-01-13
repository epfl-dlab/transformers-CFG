# Demo Guide for transformers-CFG

This guide shows you how to run various demos to see the grammar-constrained generation in action.

## Quick Start Demos

### 1. Comprehensive Demo (Recommended First!)
Shows grammar validation, token recognition, and parsing:

```bash
python demo_comprehensive.py
```

**What it demonstrates:**
- ✓ JSON grammar validation
- ✓ Prefix validation (partial strings)
- ✓ Token-level recognition with Unicode
- ✓ Grammar parsing and structure
- ✓ All refactored components working together

**Runtime:** ~5 seconds, no GPU needed

---

### 2. JSON Generation Demo
Shows constrained generation with GPT2 model:

```bash
python demo_json_generation.py
```

**What it demonstrates:**
- ✓ Loading a model (GPT2)
- ✓ Applying grammar constraints during generation
- ✓ Generating valid JSON from prompts

**Runtime:** ~30 seconds (faster with GPU)

---

## Built-in Examples

### 3. Grammar Validation
Test if a string follows a grammar:

```bash
# Valid JSON
python examples/accept.py -g examples/grammars/json.ebnf -s '{"name": "John"}' -m sentence

# Invalid JSON
python examples/accept.py -g examples/grammars/json.ebnf -s '{"invalid}' -m sentence

# Prefix validation
python examples/accept.py -g examples/grammars/json.ebnf -s '{"name"' -m prefix
```

**Parameters:**
- `-g`: Grammar file path
- `-s`: String to validate
- `-m`: Mode (`sentence` for complete, `prefix` for partial)

---

### 4. Generate JSON with Large Models
Generate JSON with Mistral-7B (requires large model download):

```bash
python examples/generate_json.py
```

**Note:** This requires downloading a 7B model (~14GB). Only run if you have:
- ✓ Good internet connection
- ✓ At least 16GB RAM or GPU memory
- ✓ ~30GB disk space

---

### 5. Other Generation Examples

Generate SMILES (chemistry):
```bash
python examples/generate_smiles.py
```

Generate PDDL (planning):
```bash
python examples/generate_pddl.py
```

Generate GeoQuery (database queries):
```bash
python examples/generate_geo_query.py
```

**Note:** These all require downloading large models.

---

## Available Grammars

See what grammars are available:

```bash
ls examples/grammars/
```

Common grammars:
- `json.ebnf` - JSON objects and arrays
- `japanese.ebnf` - Japanese characters (Unicode)
- `arithmetic.ebnf` - Simple arithmetic expressions
- `smiles.ebnf` - Chemical formulas
- `c.ebnf` - C programming language subset

---

## Running Tests

Run all tests to verify everything works:

```bash
# All tests
python -m pytest tests/ -v

# Just string validation tests (fast)
python -m pytest tests/test_accept_string/ -v

# Just GPT2 token tests (fast)
python -m pytest tests/test_accept_token_sequence/test_gpt2.py -v

# Grammar parsing tests
python -m pytest tests/test_parsing/ -v
```

---

## Quick Validation Tests

Test individual components:

```bash
# Test JSON validation
python -c "
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer

with open('examples/grammars/json.ebnf') as f:
    grammar = parse_ebnf(f.read())

recognizer = StringRecognizer(grammar.grammar_encoding, grammar.symbol_table['root'])
print('Valid JSON:', recognizer._accept_string('{\"key\": \"value\"}'))
print('Invalid JSON:', recognizer._accept_string('{invalid}'))
"
```

---

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'transformers_cfg'`
```bash
# Install the package in development mode
pip install -e .
```

**Issue:** Model download takes too long
- Use the lightweight demos (`demo_comprehensive.py`)
- Use GPT2 instead of larger models (`demo_json_generation.py`)

**Issue:** Out of memory
- Close other applications
- Use CPU instead of GPU: Set `device = "cpu"` in the script
- Use smaller models (GPT2 instead of Mistral-7B)

---

## Summary

**Start here:** `python demo_comprehensive.py`
**Then try:** `python demo_json_generation.py`
**Run tests:** `python -m pytest tests/test_accept_string/ -v`

All demos use the refactored architecture and verify that the code works correctly!
