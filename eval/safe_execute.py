"""Sandboxed execution of model-generated code against unit tests.

Model output may contain dangerous operations (os.system, file I/O, infinite
loops). Every candidate is run in an isolated subprocess with a hard timeout.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass

DEFAULT_TIMEOUT = 10  # seconds; guards against infinite loops


@dataclass
class ExecResult:
    passed: bool
    timed_out: bool
    returncode: int | None
    stderr: str
    exception_type: str | None


def safe_execute(code: str, test: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Run `code + test` in an isolated subprocess. Return True iff it exits 0."""
    return run_with_stderr(code, test, timeout).passed


def run_with_stderr(
    code: str, test: str, timeout: int = DEFAULT_TIMEOUT
) -> ExecResult:
    """Execute and capture pass/fail, timeout, and the exception type (if any)."""
    full_code = code + "\n" + test
    fname = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(full_code)
            fname = f.name
        proc = subprocess.run(
            ["python", fname],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecResult(
            passed=proc.returncode == 0,
            timed_out=False,
            returncode=proc.returncode,
            stderr=proc.stderr,
            exception_type=_extract_exception_type(proc.stderr),
        )
    except subprocess.TimeoutExpired:
        return ExecResult(
            passed=False,
            timed_out=True,
            returncode=None,
            stderr="TimeoutExpired",
            exception_type=None,
        )
    finally:
        if fname is not None and os.path.exists(fname):
            os.unlink(fname)


def runs_without_exception(code: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """True if the code alone (no test) runs to completion without raising."""
    return run_with_stderr(code, "", timeout).returncode == 0


def _extract_exception_type(stderr: str) -> str | None:
    """Pull the exception class name off the last traceback line, if present."""
    if not stderr:
        return None
    for line in reversed(stderr.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        head = line.split(":", 1)[0].strip()
        if head and head[0].isupper() and "Error" in head or head.endswith(
            ("Exception", "Error")
        ):
            return head
        return None
    return None
