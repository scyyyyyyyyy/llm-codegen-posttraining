"""Shared helpers for the data pipeline.

Prompt-pool schema (one JSON object per line):
  {
    "id": str,                 # unique, source-prefixed
    "source": str,             # "mbpp" | "kodcode" | ...
    "prompt_text": str,        # natural-language problem statement
    "entry_point": str,        # target function name
    "tests": [str, ...],       # executable assert statements (call the function)
    "reference_solution": str  # a known-correct solution (for the pre-filter only)
  }
"""

from __future__ import annotations

import ast
import json
import re


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_entry_point(code: str) -> str | None:
    """Return the name of the last top-level function def in `code`, or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    return names[-1] if names else None


_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull code out of a model response: prefer the last ```python``` block."""
    blocks = _CODE_FENCE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def normalize_code(code: str) -> str:
    """Canonicalize for near-duplicate comparison: strip comments/blank lines/space."""
    lines = []
    for line in code.splitlines():
        s = line.split("#", 1)[0].rstrip()
        if s.strip():
            lines.append(s)
    return "\n".join(lines)


def to_chat_record(prompt_text: str, entry_point: str, solution: str,
                   system: str = "You are an expert Python programmer. "
                                  "Write clean, correct code.") -> dict:
    """Format one (problem, solution) into a chat SFT record with a fixed fence."""
    user = f"Problem:\n{prompt_text}\n\nWrite the function `{entry_point}`."
    assistant = f"```python\n{solution.strip()}\n```"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
