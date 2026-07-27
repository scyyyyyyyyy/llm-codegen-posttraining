"""Error taxonomy (v2 RQ2).

Classes: correct | syntax_error | runtime_error | logic_error | timeout

Pipeline: compile check -> execute -> inspect exception type -> assertion fail.
Must be human-validated on >=50 problems and the measured accuracy reported.
Recorded at EVERY checkpoint (not just the final model) so error-type dynamics
can be plotted against training step.
"""

from __future__ import annotations

RUNTIME_EXCEPTIONS = {
    "TypeError", "IndexError", "AttributeError", "RecursionError",
    "KeyError", "ValueError", "ZeroDivisionError", "NameError",
    "OverflowError", "StopIteration",
}

ErrorLabel = str  # one of the classes above


def classify(code: str, test: str) -> ErrorLabel:
    """Return the error class for a single (code, test) pair.

    TODO:
      1. compile() -> SyntaxError => "syntax_error"
      2. sandbox.run_one(code, test)
      3. passed => "correct"; timed_out => "timeout"
      4. exception_type == "AssertionError" => "logic_error"
      5. exception_type in RUNTIME_EXCEPTIONS => "runtime_error"
    """
    raise NotImplementedError


def validate_on_sample(labeled_examples) -> float:
    """Compare classifier output against >=50 hand-labeled examples.

    Returns accuracy; report this number in the repo/blog (RQ2 credibility).
    """
    raise NotImplementedError
