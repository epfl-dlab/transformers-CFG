# transformers-CFG Architecture

This document provides a comprehensive overview of the transformers-CFG codebase architecture after the refactoring initiative.

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Module Organization](#module-organization)
5. [Design Patterns](#design-patterns)
6. [Extension Points](#extension-points)

## System Overview

transformers-CFG enables grammar-constrained text generation for large language models by masking logits according to context-free grammar rules. The system operates as a logits processor that integrates with the HuggingFace Transformers library.

### Key Capabilities

- **Grammar-Based Constraints**: Enforce EBNF grammar rules during generation
- **Incremental Processing**: Process tokens one at a time with state tracking
- **Multi-Tokenizer Support**: Works with GPT2, LLAMA, T5, and other tokenizer families
- **Efficient Masking**: Uses byte tries and caching for fast token acceptance computation

## Core Components

### 1. Grammar Module (`transformers_cfg/grammar/`)

Manages grammar encoding and parsing state using a pushdown automaton approach.

#### GrammarState
- **Responsibility**: Manages grammar encoding and provides O(1) rule lookups
- **Key Methods**: `get_rule_offset()`, `_build_rule_offsets()`
- **Data Structure**: Flattened integer array with rule offset index

#### StackManager
- **Responsibility**: Handles parsing stack operations and expansion
- **Key Methods**: `create_initial_stacks()`, `expand_stack_head()`
- **Data Structure**: Set of tuples representing parsing stacks
- **Caching**: Uses `@lru_cache` for stack expansion (32768 entries)

#### CharacterMatcher
- **Responsibility**: Matches characters/code points against grammar rules
- **Key Methods**: `accepts_code_point()`, `accepts_partial_utf8()`
- **Optimization**: LRU cache with 30000 entries for code point matching

### 2. Recognition Module (`transformers_cfg/recognition/`)

Handles batch token sequence processing and acceptance computation.

#### BatchStateProcessor
- **Responsibility**: Processes batch token sequences and updates parsing states
- **Key Methods**: `update_batch_states()`, `update_single_sequence()`
- **State Tracking**: Maintains `last_size` for incremental updates

#### TokenAcceptanceComputer
- **Responsibility**: Computes which tokens are acceptable given parsing state
- **Key Methods**: `compute_token_acceptance()`, `compute_for_single_stack()`
- **Modes**: Supports both unicode (byte-by-byte) and non-unicode (trie-based) matching

### 3. Generation Module (`transformers_cfg/generation/`)

Orchestrates grammar-constrained generation with adapters and state management.

#### GrammarConstrainedLogitsProcessor
- **Responsibility**: Main entry point for logits masking
- **Key Methods**: `mask_logits()`, `process_logits()`, `__call__()`
- **Execution Modes**:
  - `speculation`: Try most likely token first (faster)
  - `full_mask`: Compute full vocabulary acceptance (comprehensive)

#### AdapterManager
- **Responsibility**: Manages loading and invocation of generation adapters
- **Supported Adapters**: `transformers` (default), `llama_cpp_python`
- **Extension Point**: Add new adapters in `transformers_cfg/adapters/`

#### ParsingStateManager
- **Responsibility**: Manages batch parsing state initialization and updates
- **Key Methods**: `initialize_states()`, `update_states()`
- **Integration**: Coordinates with token recognizer for state updates

### 4. Tokenization Module (`transformers_cfg/tokenization/`)

Handles tokenizer-specific logic for converting tokens to bytes/characters.

#### TokenizerInspector
- **Responsibility**: Centralized tokenizer family detection
- **Key Methods**: `get_family()`, `is_deepseek_coder()`, `is_llama3_family()`
- **Families**: `gpt2`, `llama1`, `llama3`, `t5`, `byt5`, `codegen`

#### Token2ByteMapping (Abstract Base Class)
- **Responsibility**: Maps token IDs to byte sequences
- **Template Methods**: `_normalize_token_id()`, `_is_special_token()`, `_handle_special_token()`, `_get_proxy_token()`
- **Implementations**:
  - `GPT2Token2ByteMapping`: For GPT2-style tokenizers
  - `LLAMA1Token2ByteMapping`: For LLAMA1-style tokenizers (with BOS space handling)
  - `T5Token2ByteMapping`: For T5-style tokenizers (with BOS space handling)
  - `ByT5Token2ByteMapping`: For character-level tokenizers

## Data Flow

### Generation Pipeline

```
User Input
    ↓
HuggingFace Model.generate()
    ↓
GrammarConstrainedLogitsProcessor.__call__()
    ↓
ParsingStateManager.update_states()
    ↓
IncrementalTokenRecognizer.update_state_with_batch_token_seqs()
    ↓
BatchStateProcessor.update_batch_states()
    ↓
StringRecognizer._update_state_with_bytes()
    ↓
[Grammar validation happens here]
    ↓
GrammarConstrainedLogitsProcessor.mask_logits()
    ↓
TokenAcceptanceComputer.compute_token_acceptance()
    ↓
[Logits masked with -inf for invalid tokens]
    ↓
Masked Logits → Model → Next Token
```

### Grammar Validation Flow

```
Byte Sequence
    ↓
StringRecognizer._update_state_with_bytes()
    ↓
decode_utf8() → Code Points
    ↓
_update_state_with_code_points_for_all_stacks()
    ↓
For each stack:
    StackManager.expand_stack_head()
    ↓
    CharacterMatcher.accepts_code_point()
    ↓
    Update stack with next element
    ↓
New Parsing State
```

## Module Organization

```
transformers_cfg/
├── _compat.py                  # Backward compatibility utilities
├── grammar/                    # Grammar state management
│   ├── __init__.py
│   ├── grammar_state.py        # Grammar encoding & rule lookups
│   ├── stack_manager.py        # Parsing stack operations
│   └── character_matcher.py    # Character/code point matching
├── recognition/                # Token recognition components
│   ├── __init__.py
│   ├── batch_processor.py      # Batch state processing
│   └── acceptance_computer.py  # Token acceptance computation
├── generation/                 # Generation orchestration
│   ├── __init__.py
│   ├── logits_process.py       # Main logits processor
│   ├── adapter_manager.py      # Adapter loading/invocation
│   └── parsing_state_manager.py # State management
├── tokenization/               # Tokenizer-specific logic
│   ├── tokenizer_inspector.py  # Tokenizer family detection
│   ├── tokenizer.py            # TCFG tokenizer wrappers
│   ├── byte_trie.py            # Byte trie for token lookup
│   └── mapping/
│       └── token2byte.py       # Token-to-byte mappings
├── recognizer.py               # StringRecognizer (orchestrator)
├── token_grammar_recognizer.py # Token-level recognizers
├── parser.py                   # EBNF parser
└── utf8_utils.py               # UTF-8 handling utilities
```

## Design Patterns

### 1. Composition Over Inheritance

**Example**: `StringRecognizer` composes `GrammarState`, `StackManager`, and `CharacterMatcher` rather than inheriting complex behavior.

**Benefits**:
- Clear separation of concerns
- Easier testing of individual components
- Flexible component reuse

### 2. Template Method Pattern

**Example**: `Token2ByteMapping` base class provides template methods like `_normalize_token_id()` that subclasses use.

**Benefits**:
- Eliminates code duplication
- Consistent behavior across implementations
- Single source of truth for common operations

### 3. Strategy Pattern

**Example**: `GrammarConstrainedLogitsProcessor` supports different execution modes (`speculation` vs `full_mask`).

**Benefits**:
- Runtime behavior selection
- Easy to add new strategies
- Clear performance tradeoffs

### 4. Facade Pattern

**Example**: `IncrementalTokenRecognizer` provides a simple interface while delegating to `BatchStateProcessor` and `TokenAcceptanceComputer`.

**Benefits**:
- Simplified API for users
- Complex operations hidden behind clean interface
- Backward compatibility maintained

### 5. Factory Pattern

**Example**: `Token2ByteMapping.from_hf_tokenizer()` creates appropriate mapping class based on tokenizer type.

**Benefits**:
- Centralized object creation
- Easy to add new tokenizer families
- Type-safe instantiation

## Extension Points

### Adding New Tokenizer Families

1. **Update TokenizerInspector**: Add detection logic in `get_family()`
2. **Create Mapping Class**: Extend `Token2ByteMapping` for tokenizer-specific behavior
3. **Update Factory**: Add case in `Token2ByteMapping.from_hf_tokenizer()`
4. **Add to Supported List**: Update `SUPPORTED_TOKENIZERS`

### Adding New Generation Adapters

1. **Create Adapter Module**: Add file in `transformers_cfg/adapters/`
2. **Implement Adapter Function**: Takes `LogitsProcessor` and returns callable
3. **Use AdapterManager**: Automatically loaded when specified

### Customizing Grammar Rules

1. **Write EBNF Grammar**: Define rules in EBNF format
2. **Parse Grammar**: Use `parse_ebnf()` to get encoding
3. **Create Recognizer**: Instantiate `IncrementalTokenRecognizer`
4. **Apply Constraints**: Pass to `GrammarConstrainedLogitsProcessor`

## Performance Considerations

### Caching Strategy

- **Stack Expansion**: LRU cache (32768 entries) on `StackManager.expand_stack_head()`
- **Code Point Matching**: LRU cache (30000 entries) on `CharacterMatcher.accepts_code_point()`
- **Token Acceptance**: LRU cache (32768 entries) on `TokenAcceptanceComputer.compute_for_single_stack()`

### Optimization Techniques

1. **Speculation Mode**: Try most likely token first before computing full mask
2. **Byte Trie**: Efficient token lookup using prefix tree structure
3. **Early Termination**: Stop stack processing when empty stack found
4. **Batch Processing**: Process multiple sequences efficiently

### Memory Management

- Caches can accumulate memory on GPU when using CUDA
- Consider clearing caches between generation runs for long-running processes
- Use `reset()` methods to clear state

## Testing Strategy

### Unit Tests
- Grammar component tests: `tests/test_accept_string/`
- Token recognition tests: `tests/test_accept_token_sequence/`
- Parser tests: `tests/test_parsing/`

### Integration Tests
- Full generation examples in `examples/`
- Tests multiple tokenizer families
- Validates grammar constraint enforcement

### Backward Compatibility
- Deprecated names tested with warnings
- All old APIs maintained until v0.4.0

## Migration Guide

### From Pre-Refactored Version

#### Using New Component APIs

```python
# Old way (still works with deprecation warnings)
recognizer = IncrementalTokenRecognizer(grammar_str, "root", tokenizer, homomorphism=mapping)

# New way (preferred)
recognizer = IncrementalTokenRecognizer(grammar_str, "root", tokenizer, token_mapping=mapping)
```

#### Accessing Internal Components

```python
# Access new components
grammar_state = recognizer.string_recognizer.grammar_state
stack_manager = recognizer.string_recognizer.stack_manager
batch_processor = recognizer.batch_processor
acceptance_computer = recognizer.acceptance_computer
```

#### Using TokenizerInspector

```python
from transformers_cfg.tokenization.tokenizer_inspector import TokenizerInspector

inspector = TokenizerInspector(tokenizer)
family = inspector.get_family()  # Returns: "gpt2", "llama1", etc.
is_supported = inspector.is_supported()
requires_space_removal = inspector.requires_bos_space_removal()
```

## Future Improvements

### Planned Enhancements
- Additional tokenizer family support (e.g., Mistral variants)
- Optimized GPU memory usage for large vocabularies
- Parallel stack processing for better CPU utilization
- Grammar composition and inheritance

### API Evolution
- Deprecated names will be removed in v0.4.0
- New APIs designed for extensibility
- Backward compatibility maintained for one major version

## Contributing

### Code Organization Guidelines
1. **Single Responsibility**: Each class/method has one clear purpose
2. **Composition Preferred**: Use composition over inheritance
3. **Minimal Duplication**: Extract common patterns to base classes/utilities
4. **Clear Naming**: Avoid abbreviations, use descriptive names
5. **Comprehensive Documentation**: All public APIs documented

### Adding Features
1. Review architecture document
2. Identify appropriate module/component
3. Follow existing patterns (composition, template methods, etc.)
4. Add tests for new functionality
5. Update documentation

## References

- [EBNF Grammar Specification](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form)
- [Context-Free Grammars](https://en.wikipedia.org/wiki/Context-free_grammar)
- [Pushdown Automata](https://en.wikipedia.org/wiki/Pushdown_automaton)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
