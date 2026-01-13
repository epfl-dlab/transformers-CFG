# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**transformers-cfg** extends the Hugging Face Transformers library to support constrained decoding through context-free grammars (CFG). It enables grammar-constrained generation using Extended Backus-Naur Form (EBNF) grammars, similar to LlamaCPP's GBNF support but with stricter generation rules.

The library parses EBNF grammars into internal representations and uses them to mask logits during text generation, ensuring model outputs conform to specified grammar rules.

## Development Commands

### Installation
```bash
# Install in development mode with dev dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_accept_string/test_json.py

# Run tests for specific tokenizer
pytest tests/test_accept_token_sequence/test_llama.py

# Run tests with verbose output
pytest tests/ -v
```

### Code Quality
```bash
# Format code with black
black .

# Remove unused imports
autoflake --remove-all-unused-imports --in-place <file>

# Run pre-commit hooks manually
pre-commit run --all-files
```

### CLI Usage
```bash
# Check if a model is supported
transformers-cfg-cli check <model-name>

# Generate with grammar constraints
transformers-cfg-cli generate \
    -m "facebook/opt-125m" \
    -g "examples/grammars/json.ebnf" \
    -p "Generate a JSON object" \
    --max_new_tokens 50
```

## Architecture

### Core Components

#### 1. Grammar Parsing (`parser.py`)
- Parses EBNF grammar strings into internal representation
- `parse_ebnf()` converts grammar to encoded format (list of integers)
- Grammar elements: `TerminalElement` (character ranges), `ReferenceElement` (rule references), `AlternativeElements` (alternatives)
- Grammar encoding uses special markers: `END_OF_ALTERNATE_MARKER`, `END_OF_RULE_MARKER`, `END_OF_GRAMMAR_MARKER`, `REF_RULE_MARKER`

#### 2. String Recognition (`recognizer.py`)
- `StringRecognizer` class implements grammar-based string acceptance
- Uses stack-based parsing to track grammar states
- `AcceptState` holds current parsing stacks and partial UTF-8 state
- Methods: `accept_byte()`, `accept_string()`, `can_stop()`, `must_stop()`

#### 3. Token-Level Recognition (`token_grammar_recognizer.py`)
- `BaseTokenRecognizer`: Abstract base for token-level grammar recognition
- `IncrementalTokenRecognizer`: Processes tokens incrementally (default, faster)
- `NonIncrementalTokenSeqRecognizer`: Processes token sequences non-incrementally
- Maps tokens to bytes/codepoints and validates against grammar
- Key methods: `try_accept_token_id()`, `filter_vocab()`, `batch_filter_vocab()`

#### 4. Tokenization Layer (`transformers_cfg/tokenization/`)
- `ByteTrie`: Trie structure mapping byte sequences to token IDs
- `Token2ByteMapping`: Maps token IDs to byte sequences
  - Different implementations for different tokenizer families:
    - `GPT2Token2ByteMapping`: GPT2, BART, CodeGen, Qwen2, Llama-3
    - `LLAMA1Token2ByteMapping`: Llama 1/2, Gemma (uses byte proxy mapping)
    - `T5Token2ByteMapping`: T5 models
    - `ByT5Token2ByteMapping`: ByT5 models (UTF-8 byte-level)
- Handles special tokenizers like DeepSeek (GPT2-style despite inheriting from Llama)

#### 5. Logits Processing (`generation/logits_process.py`)
- `GrammarConstrainedLogitsProcessor`: Main logits processor for Transformers
- Masks invalid tokens based on current grammar state
- Two execution modes:
  - `"full_mask"` (default): Precomputes full vocabulary mask
  - `"speculation"`: Tries accepting most likely token first, falls back to full mask
- Adapter system for different frameworks:
  - `"transformers"` (default): Standard Transformers integration
  - `"llama-cpp-python"`: llama-cpp-python wrapper support
- Handles vocabulary size mismatches between tokenizer and model

### Data Flow

1. **Grammar Setup**: EBNF string → `parse_ebnf()` → Grammar encoding → `StringRecognizer`
2. **Token Recognition**: Tokenizer → `Token2ByteMapping` → `ByteTrie` → `BaseTokenRecognizer`
3. **Generation**: Model logits → `GrammarConstrainedLogitsProcessor` → Masked logits → Valid tokens only

### Key Design Patterns

- **Incremental Processing**: Grammar states are updated incrementally as tokens are generated
- **Stack-Based Parsing**: Uses stacks to track multiple possible parse paths
- **Byte-Level Mapping**: Converts tokens to bytes for grammar matching (handles multi-byte UTF-8)
- **Trie-Based Lookup**: Efficient token-to-byte mapping using trie data structure

## Important Implementation Details

### Tokenizer Support
The library supports multiple tokenizer families through `Token2ByteMapping.from_hf_tokenizer()`:
- GPT2-family: GPT2, BART, CodeGen, Qwen2, Llama-3 series, DeepSeek
- Llama-family: Llama 1/2, Gemma (use byte proxy mapping for non-ASCII)
- T5-family: T5 models
- ByT5: Byte-level models

Check `transformers_cfg/tokenization/SUPPORTED_TOKENIZERS.py` and `transformers_cfg/tokenization/mapping/token2byte.py` for full list.

### Unicode and Multi-byte Character Handling
- Grammars support Unicode characters (Chinese, Japanese, Korean, Arabic, emoji, etc.)
- Token-to-byte mapping handles multi-byte UTF-8 sequences
- `PartialUTF8` class tracks incomplete multi-byte sequences
- Grammar detection: `detect_unicode()` checks for non-ASCII characters

### EOS Token Handling
- EOS token is accepted when grammar parsing can stop (`can_stop()` returns True)
- After EOS, no more tokens are accepted (stacks are cleared)
- Special handling in `try_accept_token_id()` and `validate_and_set_eos_acceptance()`

### Grammar File Format
EBNF format, similar to LlamaCPP. Examples in `examples/grammars/`:
- Character ranges: `[a-zA-Z0-9]`
- Alternation: `"cat" | "dog"`
- Repetition: `*` (zero or more), `+` (one or more), `?` (optional)
- Repetition brackets: `{n}`, `{n,}`, `{n,m}`
- Negation: `[^a-z]` (not lowercase letters)
- Wildcards: `.` (any character)
- Grouping: `( ... )`

## Testing

### Test Structure
- `tests/test_accept_string/`: Tests grammar acceptance at string level
- `tests/test_accept_token_sequence/`: Tests with different tokenizers (GPT2, Llama, T5, etc.)
- `tests/test_parsing/`: Grammar parsing tests
- `tests/test_cli/`: CLI functionality tests

### Adding Support for New Models
1. Check tokenizer type in `transformers_cfg/tokenization/SUPPORTED_TOKENIZERS.py`
2. Add tokenizer class to `SUPPORTED_TOKENIZERS` set
3. Implement token-to-byte mapping in `token2byte.py` if needed
4. Add test file in `tests/test_accept_token_sequence/test_<model>.py`
5. Follow pattern from existing test files (e.g., `test_llama.py`)

### Test Mixin Pattern
Token sequence tests inherit from `_TestAcceptTokensMixin` which provides common test methods. Tests only need to specify the model name and optionally override fixture methods.

## Pre-commit Hooks
The repository uses pre-commit hooks (configured in `.pre-commit-config.yaml`):
- Code formatting: black
- Import cleanup: autoflake
- Standard checks: trailing whitespace, YAML syntax, large files, merge conflicts, etc.
- Tests: pytest runs on commit
- Commit message format: conventional commits (feat:, fix:, docs:, etc.)

Hooks automatically run on `git commit`. To skip hooks (not recommended): `git commit --no-verify`

## Debugging Tips

### Grammar Debugging
- See `docs/debugging_custom_grammars.md` for detailed guide
- Use `examples/accept.py` to test if a string is accepted by a grammar
- Enable verbose logging: set logging level to DEBUG

### Tokenizer Issues
- Check tokenizer charset: `transformers_cfg.tokenization.utils.get_tokenizer_charset()`
- Verify tokenizer is supported: `transformers_cfg.tokenization.utils.is_tokenizer_supported()`
- For new tokenizers, use `Token2ByteMapping.auto_infer()` (beta) or implement explicit mapping

### Performance Issues
- Use `execution_mode="speculation"` for faster generation when grammar is rarely violated
- Use smaller models for testing (e.g., facebook/opt-125m)
- See `examples/benchmarking/time_benchmarking.py` for performance testing

## Related Resources

- Original paper: "Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning" (EMNLP 2023)
- Derived from torch-grammars (Shopify) and llama-cpp
- Integrated into Text-Generation-WebUI
- JSON Schema support: See `examples/grammars/custom_json_grammars/README.md`
