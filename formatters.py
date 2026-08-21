"""
Text rendering for CLI display
"""

SEP = "=" * 64
SUBSEP = "-" * 64


def truncate(text: str, limit: int) -> str:
    """Cut text to limit chars, cutting it off with '...'"""
    if not text:
        return ""
    return text[:limit] + ("..." if len(text) > limit else "")


def format_plan_block(plan: str, target_files: list[str], limit: int = 800) -> str:
    """Plan review banner shown at the human_plan_review gate."""
    plan = plan or "(no plan generated)"
    return (
        f"\n{SEP}\n"
        f"TARGET FILES: {target_files}\n"
        f"\nINTERPRETED PLAN:\n\n"
        f"{truncate(plan, limit)}\n"
        f"{SEP}"
    )


def format_diff_block(plan: str, diff: str, plan_limit: int = 1200, diff_limit: int = 4000) -> str:
    """Plan and diff banner shown at the human_approval gate."""
    plan = plan or ""
    diff = diff or "(no diff generated)"
    return (
        f"\n{SEP}\n"
        f"CURRENT PLAN:\n\n"
        f"{truncate(plan, plan_limit)}\n"
        f"\n{SUBSEP}\n"
        f"PROPOSED DIFF:\n\n"
        f"{truncate(diff, diff_limit)}\n"
        f"{SEP}"
    )
