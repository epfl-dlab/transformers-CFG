from .token_grammar_recognizer import (
    IncrementalTokenRecognizer,
    NonIncrementalTokenSeqRecognizer,
)


# Old class name, kept for backward compatibility
IncrementalGrammarConstraint = IncrementalTokenRecognizer

NonIncrementalGrammarConstraint = NonIncrementalTokenSeqRecognizer
