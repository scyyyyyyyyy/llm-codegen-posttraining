"""Hardened sandbox for executing model-generated code (v2 §5.1).

  - subprocess isolation + wall-clock timeout
  - per-process memory cap via resource.setrlimit(RLIMIT_AS)
  - compile() precheck BEFORE spawning (skips syntax-error cases, ~30% faster)
  - a shared process pool (run_batch) reused by rejection sampling / GRPO reward

The executed script is: solution code + test source + a trailing call that
drives the test (for evalplus base tests: `check(<entry_point>)`).

NOTE on networking: true network isolation needs OS-level sandboxing (namespaces
/ containers). Here we rely on subprocess isolation + rlimit + timeout, which is
sufficient for the function-level benchmarks in this project. Do not run this on
untrusted code outside an already-isolated machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

DEFAULT_TIMEOUT = 10          # seconds
DEFAULT_MEM_LIMIT_MB = 4096   # per-process address-space cap
DEFAULT_WORKERS = 16

# rlimit is only meaningful on POSIX; skip the preexec hook elsewhere.
_POSIX = os.name == "posix"


@dataclass
class ExecResult:
    passed: bool
    timed_out: bool
    returncode: int | None
    stderr: str
    exception_type: str | None


def precheck_compiles(code: str) -> bool:
    """True if `code` parses. Cheap gate before spawning a subprocess."""
    try:
        compile(code, "<generated>", "exec")
        return True
    except SyntaxError:
        return False


def _limit_resources(mem_limit_mb: int):
    """preexec_fn factory: cap address space so runaway allocs die fast."""
    def _apply():
        import resource

        nbytes = mem_limit_mb * 1024 * 1024
        for res in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            try:
                resource.setrlimit(res, (nbytes, nbytes))
            except (ValueError, OSError):
                pass
    return _apply


def _build_script(code: str, test: str, entry_point: str | None) -> str:
    """Assemble a runnable script from solution + test source."""
    parts = [code, "", test, ""]
    if entry_point:
        # evalplus base tests define check(candidate); drive it explicitly.
        parts.append(f"check({entry_point})")
    return "\n".join(parts)


def run_one(
    code: str,
    test: str,
    entry_point: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    mem_limit_mb: int = DEFAULT_MEM_LIMIT_MB,
) -> ExecResult:
    """Execute `code + test` in an isolated subprocess with resource limits."""
    if not precheck_compiles(code):
        return ExecResult(False, False, None, "SyntaxError (precheck)", "SyntaxError")

    script = _build_script(code, test, entry_point)
    fname = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            fname = f.name
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_limit_resources(mem_limit_mb) if _POSIX else None,
        )
        return ExecResult(
            passed=proc.returncode == 0,
            timed_out=False,
            returncode=proc.returncode,
            stderr=proc.stderr,
            exception_type=_extract_exception_type(proc.stderr),
        )
    except subprocess.TimeoutExpired:
        return ExecResult(False, True, None, "TimeoutExpired", None)
    finally:
        if fname and os.path.exists(fname):
            os.unlink(fname)


def _run_one_star(args) -> ExecResult:
    return run_one(*args)


def run_batch(
    items: list[tuple],
    workers: int = DEFAULT_WORKERS,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[ExecResult]:
    """Execute many (code, test[, entry_point]) tuples concurrently, order-preserving."""
    payload = [
        (it[0], it[1], it[2] if len(it) > 2 else None, timeout) for it in items
    ]
    results: list[ExecResult | None] = [None] * len(payload)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one_star, p): i for i, p in enumerate(payload)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    return results  # type: ignore[return-value]


def passes_all(code: str, tests: list[str], entry_point: str | None = None, **kw) -> bool:
    """True iff `code` passes every test string."""
    return all(run_one(code, t, entry_point, **kw).passed for t in tests)


def runs_without_exception(code: str, entry_point: str | None = None, **kw) -> bool:
    """True iff `code` alone (no test asserts) runs to completion without raising."""
    return run_one(code, "", entry_point, **kw).returncode == 0


def _extract_exception_type(stderr: str) -> str | None:
    """Pull the exception class name off the last traceback line, if any."""
    if not stderr:
        return None
    for line in reversed(stderr.strip().splitlines()):
        line = line.strip()
        if not line or line.startswith(("File \"", "Traceback")):
            continue
        head = line.split(":", 1)[0].strip()
        if head and (head.endswith(("Error", "Exception", "Interrupt")) or head == "SystemExit"):
            return head
        return None
    return None


# Guard so ProcessPoolExecutor children don't re-run anything on import.
if __name__ == "__main__":
    _demo = textwrap.dedent(
        """
        def add(a, b):
            return a + b
        """
    )
    print(run_one(_demo, "assert add(2, 3) == 5", entry_point=None))
