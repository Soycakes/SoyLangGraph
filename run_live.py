"""
SoyLangGraph live demo — end-to-end run against real LLM APIs.
Requires: GEMINI_API_KEY env var, claude CLI installed globally.

Usage:
    python run_live.py "Add a login function to auth.py"
    python run_live.py  # prompts interactively
"""

import sys

from langgraph.types import Command

from graph import build_graph
from state import SharedBlackboard


def _initial_state(prompt: str, target_files: list[str]) -> dict:
    return {
        "original_prompt": prompt,
        "target_files": target_files,
        "layer_blackboard": SharedBlackboard(),
        "coworker_findings": [],
        "proposed_diff": None,
        "critique_feedback": None,
        "syntax_valid": False,
        "unresolved_imports": [],
        "redteam_fatal_flaws": [],
        "diff_line_count": 0,
        "critique_iteration_count": 0,
        "syntax_retry_count": 0,
        "fanout_depth": 0,
        "interrupt_reason": None,
        "current_plan": None,
    }


def main() -> None:
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Task: ").strip()
        if not prompt:
            print("No task provided.")
            return

    raw_files = input("Target file(s) [main.py]: ").strip() or "main.py"
    target_files = [f.strip() for f in raw_files.split(",")]

    g = build_graph()
    config = {"configurable": {"thread_id": "live-demo"}}

    print(f"\n[SoyLangGraph] Starting: {prompt!r}")
    print(f"[SoyLangGraph] Targets:  {target_files}\n")

    g.invoke(_initial_state(prompt, target_files), config=config)

    while True:
        snap = g.get_state(config)
        if not snap.next:
            print("\n[SoyLangGraph] Complete.")
            break

        node = snap.next[0]

        if node == "human_approval":
            diff = snap.values.get("proposed_diff") or "(no diff generated)"
            plan = snap.values.get("current_plan") or ""
            print("\n" + "═" * 64)
            print("CURRENT PLAN:\n")
            print(plan[:1200] + ("…" if len(plan) > 1200 else ""))
            print("\n" + "─" * 64)
            print("PROPOSED DIFF:\n")
            print(diff[:4000] + ("…" if len(diff) > 4000 else ""))
            print("═" * 64)
            answer = input("\nApprove this diff? [y/N]: ").strip().lower()
            g.invoke(Command(resume=(answer == "y")), config=config)

        elif node == "interrupt_human_escalation":
            reason = snap.values.get("interrupt_reason") or "circuit_breaker"
            print(f"\n[SoyLangGraph] Escalation — human input required: {reason}")
            input("Press Enter to acknowledge and end this session...")
            g.invoke(Command(resume=True), config=config)
            break

        else:
            print(f"\n[SoyLangGraph] Unexpected pause at node: {node!r}")
            break


if __name__ == "__main__":
    main()
