# Debugging custom grammars

This document provides best practices for debugging custom grammars when using the `transformers_cfg` library. It offers strategies to help identify and resolve common issues during grammar creation or modification.

## Table of contents

- [Introduction](#introduction)
- [Syntax highlighting](#syntax-highlighting)
- [EBNF and variants](#ebnf-and-variants)
- [Check parse](#check-parse)
- [Test with input](#test-with-input)
- [Debug mode](#debug-mode)
- [Tips and tricks](#tips-and-tricks)
    - [Incremental development](#incremental-development)
    - [Isolate grammar components](#isolate-grammar-components)
    - [Test with language model](#test-with-language-model)

## Introduction

Context-free grammars (CFGs) involve complex syntax and semantics. This guide outlines strategies and tools to help debug custom grammars effectively. The `transformers_cfg` library uses EBNF notation for grammar definition and aligns with the grammar module of [llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master/grammars). For an introduction to EBNF, refer to the [llama.cpp documentation](https://github.com/ggerganov/llama.cpp/tree/master/grammars), where EBNF is referred to as `gbnf` for its integration with the project

## Syntax highlighting

The Visual Studio Code extension EBNF offers syntax highlighting for EBNF grammars.

<p align="center">
  <img src="assets/screenshots/vscode_ebnf_syntax_highlight.png" alt="EBNF syntax highlighting" width="100%">
</p>
<p align="center"><em>Figure 1: EBNF syntax highlighting</em></p>

For IDEs that use the Open VSX marketplace, such as Trae, the [W3C EBNF extension](https://open-vsx.org/extension/mfederczuk/w3c-ebnf) provides similar functionality. In JetBrains IDEs, the [Context Free Grammar plugin](https://plugins.jetbrains.com/plugin/10162-context-free-grammar) is available.

These extensions are third-party tools and are not affiliated with `transformers_cfg`. Use them responsibly and report any alternative suggestions.

## EBNF and variants

EBNF is a notation with several variants, each featuring slightly different syntax while preserving the underlying semantics.

The two major variants are:

- [ISO/IEC 14977](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form): The original standard for EBNF.
- [W3C EBNF](https://www.w3.org/TR/REC-xml/#sec-notation): The variant used in the W3C XML specification.

The EBNF variant in `transformers_cfg` aligns mostly with the W3C version, with one exception: the negation operator (`^`) is not yet supported but will be added in a future update.

## Check parse

To verify that an EBNF grammar is correct, use the `transformers_cfg/parser::parse_ebnf` function. If Graphviz is installed, generate a parse tree by adding the `--graph` option.

```terminal
python -m transformers_cfg.parser --grammar-file examples/grammars/your_grammar.ebnf
```

Example output for a JSON grammar is:

```terminal
Grammar Rules:
<0>root_2 ::= <2>jp-char <4>root_2 | <8>jp-char
<12>root_4 ::= <14>jp-char <16>root_4 | <20>jp-char
<24>root_3 ::= <26>[ -  -
-
] <33>root_4
<37>root_5 ::= <39>root_3 <41>root_5 |
<47>root ::= <49>root_2 <51>root_5
<55>jp-char ::= <57>hiragana | <61>katakana | <65>punctuation | <69>cjk
<73>hiragana ::= <75>[ぁ-ゟ]
<80>katakana ::= <82>[ァ-ヿ]
<87>punctuation ::= <89>[、-〾]
<94>cjk ::= <96>[一-鿿]

Grammar Hex representation:
0002 0005 0001 0001 0001 0002 0000 0003 0001 0001 0000 0000 0004 0005 0001 0001 0001 0004 0000 0003 0001 0001 0000 0000 0003 000a 0006 0020 0020 0009 0009 000a 000a 0001 0004 0000 0000 0005 0005 0001 0003 0001 0005 0000 0001 0000 0000 0000 0005 0001 0002 0001 0005 0000 0000 0001 0003 0001 0006 0000 0003 0001 0007 0000 0003 0001 0008 0000 0003 0001 0009 0000 0000 0006 0004 0002 3041 309f 0000 0000 0007 0004 0002 30a1 30ff 0000 0000 0008 0004 0002 3001 303e 0000 0000 0009 0004 0002 4e00 9fff 0000 0000 ffff

Rules Decimal representation:
<2> [[5, 1, 1, 1, 2, 0], [3, 1, 1, 0]]
<4> [[5, 1, 1, 1, 4, 0], [3, 1, 1, 0]]
<3> [[10, 6, 32, 32, 9, 9, 10, 10, 1, 4, 0]]
<5> [[5, 1, 3, 1, 5, 0], [1, 0]]
<0> [[5, 1, 2, 1, 5, 0]]
<1> [[3, 1, 6, 0], [3, 1, 7, 0], [3, 1, 8, 0], [3, 1, 9, 0]]
<6> [[4, 2, 12353, 12447, 0]]
<7> [[4, 2, 12449, 12543, 0]]
<8> [[4, 2, 12289, 12350, 0]]
<9> [[4, 2, 19968, 40959, 0]]
symbol_ids:
{'root': 0, 'jp-char': 1, 'root_2': 2, 'root_3': 3, 'root_4': 4, 'root_5': 5, 'hiragana': 6, 'katakana': 7, 'punctuation': 8, 'cjk': 9}
```

A successful parse confirms the grammar is syntactically correct.

<p align="center">
  <img src="assets/plots/arithmetic_grammar_viz.png" alt="Visualization of arithmetic grammar" width="50%">
</p>
<p align="center"><em>Figure 2: Graph visualization of the arithmetic grammar</em></p>

## Test with input

After verifying that the grammar can be parsed, test it with a simple input to confirm the expected output. The following script demonstrates this process:

```python
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer

with open("examples/grammars/json.ebnf", "r") as file:
    input_text = file.read()
parsed_grammar = parse_ebnf(input_text)

start_rule_id = parsed_grammar.symbol_table["root"]
recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

# Test the grammar with a simple input.
json_input = '{"foo": "bar", "baz": "bat"}'
is_accepted = recognizer._accept_prefix(json_input)
print(is_accepted)
```

If the script prints `True`, the grammar recognizes the input string. A result of `False` indicates that the input is not fully recognized. To identify the failure point, try testing with a partial input:

```python
json_input = '{"foo": "bar"'
is_accepted = recognizer._accept_prefix(json_input)
print(is_accepted)
```

To verify if the input string is complete, use the `_accept_string` method, which returns `True` for a complete string and `False` otherwise.

## Debug mode

Enable debug mode to observe the parsing process in detail by setting the environment variable:

```bash
export TCFG_LOG_LEVEL=DEBUG
```

The output will log each accepted code point. For example:

```terminal
DEBUG:root:code point [123] corresponding to { is accepted
DEBUG:root:code point [123, 34] corresponding to " is accepted
...
DEBUG:root:code point [123, 34, 102, 111, 111, 34, 58, 32, 34, 98, 97, 116, 34, 125] corresponding to } is accepted
```

This log assists in identifying where the parser accepts or rejects input characters.

## Tips and tricks

### Incremental development

Begin with a minimal grammar rule and gradually add more rules. This incremental approach simplifies error detection as the grammar evolves.

### Isolate grammar components

If the grammar does not behave as expected, isolate individual components to determine the source of the issue. Remove or comment out parts of the grammar and reintroduce them gradually until the problem is identified.

### Test with language model

Once the grammar is confirmed to be correct, remaining issues likely pertain to other aspects of the system. Testing with a language model is important at this stage, although it falls outside the scope of grammar verification.
