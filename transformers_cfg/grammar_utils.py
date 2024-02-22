from .token_grammar_recognizer import IncrementalTokenRecognizer


# Old class name, kept for backward compatibility
class IncrementalGrammarConstraint(IncrementalTokenRecognizer):
    def __init__(self, *args, **kwargs):
        # import warnings
        # warnings.warn(
        #     "IncrementalGrammarConstraint is deprecated and will be removed in a future version. "
        #     "Please use IncrementalLLMGrammarRecognizer instead.",
        #     DeprecationWarning, stacklevel=2
        # )
        super().__init__(*args, **kwargs)
