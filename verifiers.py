"""
Deterministic verifiers — zero LLM cost, run post-Layer 4 before human gate.
"""

import ast
import re
import textwrap


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```diff, ```python, ``` etc.) from LLM stdout."""
    return re.sub(r"^\s*```[\w]*\n?|^\s*```$", "", text, flags=re.MULTILINE).strip()


def count_diff_lines(diff: str) -> int:
    """Count changed lines in a unified diff (additions + deletions, excluding file headers)."""
    return sum(
        1 for line in diff.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )


def extract_diff_additions(diff: str) -> str:
    """Return added lines from a unified diff as a single source string (+ prefix stripped)."""
    return "\n".join(
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def validate_python_syntax(source: str) -> tuple[bool, str]:
    """Try ast.parse() on source. Returns (True, '') on success, (False, error_msg) on failure.
    ponytail: validates extracted additions only — not full file context.
    Full-file validation (apply diff → parse whole file) belongs in Phase 3.
    dedent strips common leading whitespace so additions inside indented scopes parse correctly.
    """
    try:
        ast.parse(textwrap.dedent(source))
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.msg} (line {exc.lineno})"
