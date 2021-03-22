import pandas as pd
from .model import QUERIES, RANKED_DOCS, RANKED_DOCS_FEATURES

# Define the strictness level of transformer type checking
# 0 : Only fail if validation causes error
# 1 : Give warning when a suitable minimum_input/minimum_output not found for a transformer
# 2 : Raise error when a suitable minimum_input/minimum_output not found for a transformer
TYPE_SAFETY_LEVEL = 0

TRANSFORMER_FAMILY = {
    'queryrewrite': {
        'minimal_input': QUERIES,
        'minimal_output': QUERIES
    },
    'retrieval': {
        'minimal_input': QUERIES,
        'minimal_output': RANKED_DOCS,
    },
    'queryexpansion': {
        'minimal_input': RANKED_DOCS,
        'minimal_output': QUERIES,
    },
    'reranking': {
        'minimal_input': RANKED_DOCS,
        'minimal_output': RANKED_DOCS,
    },
    'featurescoring': {
        'minimal_input': QUERIES,
        'minimal_output': RANKED_DOCS_FEATURES,
    }
}


class PipelineError(Exception):
    """
    Exception raised when an error occurs in a pipeline

    Attributes:
        t1: The transformer that has failed validation
        bad_input: The input columns that caused the failure
        t2: Transformer whose output caused the failure
        message: Explanation of error
    """

    def __init__(self, t1, bad_input, t2=None):
        self.t1 = t1
        if isinstance(bad_input, pd.DataFrame):
            bad_input = list(bad_input.columns)
        self.bad_input = bad_input
        self.t2 = t2
        self.message = self._generate_error_message(self.t1, bad_input, self.t2)
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def _generate_error_message(self, t1, bad_input, t2=None):
        msg = "Occurs at Transformer %s\n" \
              "Transformer requires columns %s to perform a valid transformation\n" \
              "However, only received columns %s as input in pipeline " % (repr(t1), str(t1.minimal_input), str(bad_input))
        if t2:
            msg += "\n Transformer receives these columns from %s" % repr(t2)
        return msg


class ValidationError(Exception):
    """
    Exception raised when validation cannot occur

    Attributes:
        t1: The transformer that validation does not work for
        input_to_validate: The input that the transformer wished to validate
        message: Explanation of error, with hint to how it might be resolved
    """

    def __init__(self, t1, input_to_validate):
        self.t1 = t1
        if isinstance(input_to_validate, pd.DataFrame):
            input_to_validate = list(input_to_validate.columns)
        self.message = self._generate_error_message(self.t1, input_to_validate)
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def _generate_error_message(self, t1, input_to_validate):
        msg = ""
        if not t1.minimal_input:
            msg += "Minimal input, "
        if not t1.minimal_output:
            msg += "Minimal output, "
        msg += "not defined for transformer %s, therefore pipeline cannot be validated.\n" \
               "Minimal input Hint - Transformer receives columns %s" % (repr(t1), str(input_to_validate))
        return msg