"""Error taxonomy for generated code: the project's core analysis tool.

Three failure classes:
  - syntax_error : code does not parse (compile raises SyntaxError)
  - runtime_error: parses and runs but raises an exception
  - logic_error  : runs cleanly but a test assertion fails (wrong output)

Plus `correct` and `timeout`. The progression of these classes across
SFT / DPO stages is a central finding of the writeup.
"""

from __future__ import annotations

from .safe_execute import run_with_stderr

RUNTIME_EXCEPTIONS = {
    "TypeError",
    "IndexError",
    "AttributeError",
    "RecursionError",
    "KeyError",
    "ValueError",
    "ZeroDivisionError",
    "NameError",
    "OverflowError",
    "StopIteration",
}


def classify_error(code: str, test: str) -> str:
    """Return one of: correct | syntax_error | runtime_error | logic_error | timeout."""
    # 1. Does it even parse?
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError:
        return "syntax_error"

    # 2. Run against the test.
    result = run_with_stderr(code, test)
    if result.passed:
        return "correct"
    if result.timed_out:
        return "timeout"

    # 3. Distinguish runtime exceptions from assertion (logic) failures.
    exc = result.exception_type
    if exc == "AssertionError":
        return "logic_error"
    if exc in RUNTIME_EXCEPTIONS:
        return "runtime_error"
    # Unknown exception type: treat as runtime to be safe.
    return "runtime_error" if exc else "logic_error"
