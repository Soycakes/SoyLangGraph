"""
Syntax and diff checks that run before the human approval gate.
"""

import ast
import re
import textwrap


def strip_markdown_fences(text: str) -> str:
    """Extracts code from markdown fences if present, returns the text stripped otherwise."""
    m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def count_diff_lines(diff: str) -> int:
    """Counts added and deleted lines in a unified diff, ignoring file headers."""
    return sum(
        1 for line in diff.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )


def extract_diff_additions(diff: str) -> str:
    """Pulls added lines out of a unified diff, stripping the leading + prefix."""
    return "\n".join(
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def validate_python_syntax(source: str) -> tuple[bool, str]:
    """Parses source with ast.parse. Returns (True, '') or (False, error message).
    dedent handles additions extracted from inside indented blocks.
    """
    try:
        ast.parse(textwrap.dedent(source))
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.msg} (line {exc.lineno})"
