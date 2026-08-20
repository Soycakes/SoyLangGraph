"""
Unit tests for deterministic verifiers — no LLM, no graph, no IO.
"""

import pytest

from verifiers import (
    count_diff_lines,
    extract_diff_additions,
    strip_markdown_fences,
    validate_python_syntax,
)

SAMPLE_DIFF = (
    "--- a/auth/login.py\n"
    "+++ b/auth/login.py\n"
    "@@ -1,3 +1,5 @@\n"
    " import os\n"
    "-old_line = 1\n"
    "+new_line = 1\n"
    "+extra_line = 2\n"
    " unchanged = True\n"
)


# ── strip_markdown_fences ─────────────────────────────────────────────────────

class TestStripMarkdownFences:
    def test_strips_diff_fence(self):
        text = "```diff\n--- a/f.py\n+++ b/f.py\n```"
        assert strip_markdown_fences(text) == "--- a/f.py\n+++ b/f.py"

    def test_strips_python_fence(self):
        text = "```python\nx = 1\n```"
        assert strip_markdown_fences(text) == "x = 1"

    def test_strips_bare_fence(self):
        text = "```\nsome content\n```"
        assert strip_markdown_fences(text) == "some content"

    def test_no_fence_unchanged(self):
        diff = "--- a/f.py\n+++ b/f.py\n+x = 1\n"
        assert strip_markdown_fences(diff) == diff.strip()

    def test_empty_string(self):
        assert strip_markdown_fences("") == ""

    def test_strips_fence_preserves_inner_content(self):
        text = "```diff\n+added line\n-removed line\n```"
        result = strip_markdown_fences(text)
        assert "+added line" in result
        assert "-removed line" in result
        assert "```" not in result


# ── count_diff_lines ──────────────────────────────────────────────────────────

class TestCountDiffLines:
    def test_counts_additions_and_deletions(self):
        # SAMPLE_DIFF has 1 deletion (-old_line) and 2 additions (+new_line, +extra_line)
        assert count_diff_lines(SAMPLE_DIFF) == 3

    def test_excludes_file_headers(self):
        diff = "--- a/f.py\n+++ b/f.py\n+actual change\n"
        assert count_diff_lines(diff) == 1

    def test_excludes_context_lines(self):
        diff = "@@ -1,3 +1,3 @@\n unchanged\n also unchanged\n"
        assert count_diff_lines(diff) == 0

    def test_empty_diff(self):
        assert count_diff_lines("") == 0

    def test_only_additions(self):
        diff = "+line one\n+line two\n+line three\n"
        assert count_diff_lines(diff) == 3

    def test_only_deletions(self):
        diff = "-line one\n-line two\n"
        assert count_diff_lines(diff) == 2


# ── extract_diff_additions ────────────────────────────────────────────────────

class TestExtractDiffAdditions:
    def test_extracts_added_lines(self):
        result = extract_diff_additions(SAMPLE_DIFF)
        assert "new_line = 1" in result
        assert "extra_line = 2" in result

    def test_excludes_deletions(self):
        result = extract_diff_additions(SAMPLE_DIFF)
        assert "old_line" not in result

    def test_excludes_file_headers(self):
        result = extract_diff_additions(SAMPLE_DIFF)
        assert "+++" not in result
        assert "a/auth" not in result

    def test_excludes_context_lines(self):
        result = extract_diff_additions(SAMPLE_DIFF)
        assert "unchanged = True" not in result

    def test_empty_diff_returns_empty_string(self):
        assert extract_diff_additions("") == ""

    def test_strips_leading_plus(self):
        diff = "+++ b/f.py\n+x = 1\n"
        result = extract_diff_additions(diff)
        assert result == "x = 1"


# ── validate_python_syntax ────────────────────────────────────────────────────

class TestValidatePythonSyntax:
    def test_valid_assignment(self):
        valid, msg = validate_python_syntax("x = 1")
        assert valid is True
        assert msg == ""

    def test_valid_function(self):
        valid, msg = validate_python_syntax("def foo(a, b):\n    return a + b\n")
        assert valid is True

    def test_valid_empty_string(self):
        valid, _ = validate_python_syntax("")
        assert valid is True

    def test_invalid_syntax_returns_false(self):
        valid, msg = validate_python_syntax("def foo(:\n    pass")
        assert valid is False
        assert msg != ""

    def test_error_message_contains_line_number(self):
        _, msg = validate_python_syntax("x = (\n  1 +\n")
        assert "line" in msg

    def test_valid_multiline(self):
        source = "x = 1\ny = x + 2\nz = x * y\n"
        valid, _ = validate_python_syntax(source)
        assert valid is True

    def test_invalid_indentation(self):
        valid, msg = validate_python_syntax("if True:\nno_indent = 1")
        assert valid is False

    def test_indented_additions_from_function_body(self):
        """Additions inside a function body carry leading whitespace — dedent must handle them."""
        source = "    x = 1\n    return x"  # as extracted from a diff inside a def block
        valid, msg = validate_python_syntax(source)
        assert valid is True, f"Expected valid after dedent, got: {msg}"

    def test_return_statement_in_partial_additions_passes(self):
        """ast.parse in exec mode does not enforce return-outside-function (that's a compile-time check).
        Partial additions containing return statements are not false-positively rejected in Python 3.12+.
        """
        valid, _ = validate_python_syntax("return x")
        assert valid is True


# ── end-to-end: strip → extract → validate ───────────────────────────────────

class TestVerifierPipeline:
    def test_fenced_diff_with_valid_python(self):
        fenced = "```diff\n--- a/f.py\n+++ b/f.py\n+x = 1\n+y = 2\n```"
        diff = strip_markdown_fences(fenced)
        additions = extract_diff_additions(diff)
        valid, _ = validate_python_syntax(additions)
        assert valid is True

    def test_fenced_diff_with_invalid_python(self):
        fenced = "```diff\n--- a/f.py\n+++ b/f.py\n+def broken(:\n+    pass\n```"
        diff = strip_markdown_fences(fenced)
        additions = extract_diff_additions(diff)
        valid, _ = validate_python_syntax(additions)
        assert valid is False

    def test_line_count_after_fence_strip(self):
        fenced = "```diff\n+a\n+b\n-c\n```"
        assert count_diff_lines(strip_markdown_fences(fenced)) == 3
