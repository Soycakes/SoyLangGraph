"""
end to end run against real LLM APIs.
Requires GEMINI_API_KEY env var.

Usage:
    python run_live.py "YOUR PROMPT HERE"
    python run_live.py
"""

import argparse
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from langgraph.types import Command

from formatters import format_diff_block, format_plan_block
from graph import build_graph
from state import SharedBlackboard


def _initial_state(prompt: str, target_files: list[str], execution_mode: str = "interactive") -> dict:
    return {
        "original_prompt": prompt,
        "target_files": target_files,
        "layer_blackboard": SharedBlackboard(),
        "coworker_findings": [],
        "proposed_content": None,
        "proposed_diff": None,
        "critique_feedback": None,
        "syntax_valid": False,
        "unresolved_imports": [],
        "diff_line_count": 0,
        "critique_iteration_count": 0,
        "syntax_retry_count": 0,
        "fanout_depth": 0,
        "interrupt_reason": None,
        "current_plan": None,
        "execution_mode": execution_mode,
        "workspace_context": None,
        "plan_feedback": None,
        "plan_approved": None,
    }


def _run(g, input, config) -> None:
    """Streams the graph and prints each node as it completes."""
    t = time.time()
    for chunk in g.stream(input, config=config, stream_mode="updates"):
        elapsed = time.time() - t
        for node_name in chunk:
            print(f"  >> {node_name} ({elapsed:.1f}s)")
        t = time.time()


def main() -> None:
    parser = argparse.ArgumentParser(description="SoyLangGraph live demo")
    parser.add_argument("task", nargs="*", help="Task description")
    parser.add_argument("--auto", action="store_true", help="Skip human gates (auto mode)")
    args = parser.parse_args()

    if args.task:
        prompt = " ".join(args.task)
    else:
        prompt = input("Task: ").strip()
        if not prompt:
            print("No task provided.")
            return

    execution_mode = "auto" if args.auto else "interactive"
    g = build_graph()
    config = {"configurable": {"thread_id": f"live-{int(time.time())}"}}

    print(f"\n[SoyLangGraph] Starting: {prompt!r}\n")

    _run(g, _initial_state(prompt, [], execution_mode), config)

    while True:
        snap = g.get_state(config)
        if not snap.next:
            print("\n[SoyLangGraph] Complete.")
            break

        node = snap.next[0]

        if node == "human_plan_review":
            plan = snap.values.get("current_plan")
            files = snap.values.get("target_files") or []
            print(format_plan_block(plan, files))
            answer = input("\nProceed? [Y/n/or type a correction]: ").strip()
            if answer.lower() in ("", "y", "yes"):
                resume = True
            elif answer.lower() in ("n", "no"):
                resume = False
            else:
                resume = answer
            _run(g, Command(resume=resume), config)

        elif node == "human_approval":
            diff = snap.values.get("proposed_diff")
            plan = snap.values.get("current_plan")
            print(format_diff_block(plan, diff))
            answer = input("\nApprove this diff? [y/N]: ").strip().lower()
            _run(g, Command(resume=(answer == "y")), config)

        elif node == "interrupt_human_escalation":
            reason = snap.values.get("interrupt_reason") or "max retries exceeded"
            print(f"\n[SoyLangGraph] Stopped: {reason}")
            input("Press Enter to exit...")
            _run(g, Command(resume=True), config)
            break

        else:
            print(f"\n[SoyLangGraph] Unexpected pause at node: {node!r}")
            break


if __name__ == "__main__":
    main()
