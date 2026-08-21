"""
Orchestration graph.
Defines the nodes and routing logic for the state machine.
"""

import difflib
import glob
from typing import Optional
import os
import re

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send, interrupt

from config import (
    COWORKER_PROMPTS,
    GEMINI_MODEL,
    L1_SYSTEM_PROMPT,
    L4_SYSTEM_PROMPT,
    get_gemini_client,
)

from state import CoworkerFinding, CoworkerInput, SharedBlackboard, SoyGraphState
from verifiers import count_diff_lines, extract_diff_additions, strip_markdown_fences, validate_python_syntax

# Config
MAX_CRITIQUE_CYCLES = 3
MAX_SYNTAX_RETRIES = 2


# Helpers

def _blackboard_field(state: SoyGraphState, field: str, default):
    """Reads a field from the blackboard, which MemorySaver may deserialize as a plain dict."""
    bb = state["layer_blackboard"]
    if isinstance(bb, dict):
        return bb.get(field, default)
    return getattr(bb, field, default)


def _gemini_generate(client, prompt: str, system: str) -> str:
    """Single call wrapper around client.models.generate_content."""
    from google.genai import types
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return (response.text or "").strip()


def _extract_bullets(text: str) -> list[str]:
    return [
        line.lstrip("-•* ").strip()
        for line in text.splitlines()
        if line.strip().startswith(("-", "•", "*")) and line.strip().lstrip("-•* ")
    ]


def _parse_l1_response(text: str) -> dict:
    """Extract PLAN and FILES from L1 response. Handles markdown bolding Gemini sometimes adds."""
    result = {}
    plan_match = re.search(r"(?:\*\*)?PLAN:?(?:\*\*)?\s*(.*?)(?=\nFILES:|\nCOMPLEXITY:|$)", text, re.IGNORECASE | re.DOTALL)
    result["current_plan"] = plan_match.group(1).strip() if plan_match else text

    files_match = re.search(r"(?:\*\*)?FILES:?(?:\*\*)?\s*(.*?)(?=\nCOMPLEXITY:|$)", text, re.IGNORECASE | re.DOTALL)
    if files_match:
        files = [f.strip() for f in files_match.group(1).strip().split(",") if f.strip()]
        if files:
            result["target_files"] = files

    return result


def _compute_diff(filename: str, old_content: Optional[str], new_content: str) -> str:
    """Produce a diff from old file content (or new files)"""
    old_lines = (old_content or "").splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    fromfile = f"a/{filename}" if old_content else "/dev/null"
    return "".join(difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=f"b/{filename}"))


def _parse_coworker_response(text: str, bias: str) -> dict:
    """Parse coworker LLM output into a CoworkerFinding, or return {} on NO_ISSUES.

    Note : Return {} (omitting coworker_findings), an empty list [] would wipe sibling coworker results.
    """
    if "NO_ISSUES" in text.upper():
        return {}
    return {
        "coworker_findings": [
            CoworkerFinding(
                agent_id=f"coworker-{bias}-live",
                perspective_bias=bias,
                assessment=text,
                suggested_modifications=_extract_bullets(text),
            )
        ]
    }


# Layer 0: Workspace scanner

_SCAN_SKIP = {".venv", "venv", "__pycache__", ".git", ".pytest_cache", "node_modules"}

def workspace_scan(state: SoyGraphState) -> dict:
    """Reads current file contents and repo layout before L1 runs."""
    file_contexts = {}
    for f in state.get("target_files", []):
        if os.path.exists(f):
            try:
                with open(f, encoding="utf-8", errors="replace") as fh:
                    file_contexts[f] = fh.read(8000)
            except OSError:
                file_contexts[f] = "[unreadable]"
        else:
            file_contexts[f] = None  # None = new file

    repo_tree = sorted(
        e for e in glob.glob("*") + glob.glob("*/*")
        if not any(part in _SCAN_SKIP for part in e.replace("\\", "/").split("/"))
    )[:60]

    deps = ""
    for dep_file in ("pyproject.toml", "requirements.txt", "package.json"):
        if os.path.exists(dep_file):
            try:
                with open(dep_file, encoding="utf-8", errors="replace") as fh:
                    deps = f"[{dep_file}]\n{fh.read(2000)}"
                break
            except OSError:
                pass

    return {"workspace_context": {"file_contexts": file_contexts, "repo_tree": repo_tree, "deps": deps}}


# Layer 1: Context Analyst

def layer1_context_check(state: SoyGraphState) -> dict:
    """Analyzes the task and workspace context, produces an initial plan."""
    workspace = state.get("workspace_context") or {}
    file_contexts = workspace.get("file_contexts") or {}
    repo_tree = workspace.get("repo_tree") or []
    deps = workspace.get("deps") or ""

    file_section = ""
    for f, content in file_contexts.items():
        if content is None:
            file_section += f"\n{f}: NEW FILE (does not exist yet)\n"
        else:
            file_section += f"\n{f} (existing, first 2000 chars):\n{content[:2000]}\n"

    prompt = state["original_prompt"]
    plan_feedback = state.get("plan_feedback")
    if plan_feedback:
        prompt += f"\n\nPrevious plan was rejected. User correction: {plan_feedback}"
    if file_section:
        prompt += f"\n\nWorkspace context:{file_section}"
    if repo_tree:
        prompt += f"\n\nRepo structure: {', '.join(repo_tree[:20])}"
    if deps:
        prompt += f"\n\nDependencies:\n{deps}"

    client = get_gemini_client()
    if client is None:
        stub = f"[L1 stub] analyse: {state['original_prompt'][:80]}"
        if plan_feedback:
            stub += f" (revised: {plan_feedback[:40]})"
        return {"current_plan": stub}
    text = _gemini_generate(client, prompt, L1_SYSTEM_PROMPT)
    return _parse_l1_response(text)


# Plan gate

def human_plan_review(state: SoyGraphState) -> dict:
    """Checkpoint before fanout: user confirms or steers the L1 plan before coworkers run.
    Resume: True = proceed, False = retry L1 silently, str = retry L1 with correction text.
    """
    feedback = interrupt({
        "current_plan": state.get("current_plan"),
        "target_files": state.get("target_files"),
        "reason": "plan_approval",
    })
    if feedback is True:
        return {"plan_approved": True, "plan_feedback": None}
    if isinstance(feedback, str) and feedback:
        return {"plan_approved": False, "plan_feedback": feedback}
    return {"plan_approved": False, "plan_feedback": None}


# Layer 2: Architect + Coworker Fanout

def layer2_architect(state: SoyGraphState) -> dict:
    """Expands the plan into a spec and decides whether to fan out to coworkers."""
    return {
        "current_plan": state.get("current_plan") or "[L2 stub] architecture plan",
        "fanout_depth": 1,
    }


def coworker_agent(state: CoworkerInput) -> dict:
    """Reviews the plan from a specific perspective (minimalist/cynic/optimizer).
    Returns {} when no findings - an empty list would wipe sibling results via the reducer.
    """
    bias = state.get("bias", "minimalist")
    client = get_gemini_client()
    if client is None:
        return {
            "coworker_findings": [
                CoworkerFinding(
                    agent_id=f"coworker-{bias}",
                    perspective_bias=bias,
                    assessment=f"[{bias} stub] no issues found",
                    suggested_modifications=[],
                )
            ]
        }
    prompt = (
        f"Plan to review:\n{state.get('plan_to_review', '')}\n\n"
        f"System constraints:\n{state.get('system_constraints', [])}"
    )
    text = _gemini_generate(client, prompt, COWORKER_PROMPTS[bias])
    return _parse_coworker_response(text, bias)


def synthesize_plan(state: SoyGraphState) -> dict:
    """Merges coworker feedback into the plan and clears the findings buffer."""
    findings = state.get("coworker_findings", [])
    base_plan = state.get("current_plan") or ""
    if findings:
        feedback = "\n\n".join(
            f"[{f.perspective_bias}]\n{f.assessment}" for f in findings
        )
        plan = f"{base_plan}\n\nCoworker review:\n{feedback}"
    else:
        plan = base_plan
    return {
        "current_plan": plan,
        "coworker_findings": [],  # reset sentinel: reduce_findings treats [] as replace
    }


# Layer 3: Technical Critique

def layer3_critique(state: SoyGraphState) -> dict:
    """Technical critic - stub, always passes."""
    return {"critique_feedback": "PASS"}


def handle_critique_failure(state: SoyGraphState) -> dict:
    """Increments critique_iteration_count before routing back to L2."""
    return {"critique_iteration_count": state.get("critique_iteration_count", 0) + 1}


# Layer 4: Lead Synthesizer

def layer4_synthesize(state: SoyGraphState) -> dict:
    """Generates the full file content and computes a diff against the current version."""
    target_files = state.get("target_files") or []
    if not target_files:
        return {"proposed_content": None, "proposed_diff": None, "diff_line_count": 0,
                "critique_feedback": "Layer 1 did not identify a target file"}
    primary = target_files[0]
    workspace = state.get("workspace_context") or {}
    old_content: Optional[str] = (workspace.get("file_contexts") or {}).get(primary)
    if old_content is None and os.path.exists(primary):
        with open(primary, encoding="utf-8", errors="replace") as _fh:
            old_content = _fh.read()

    client = get_gemini_client()
    if client is None:
        stub_content = "# stub\n"
        diff = _compute_diff(primary, old_content, stub_content)
        return {"proposed_content": stub_content, "proposed_diff": diff, "diff_line_count": count_diff_lines(diff)}

    prior_error = state.get("critique_feedback")
    prompt = (
        f"Write the complete implementation of `{primary}`.\n\n"
        f"TASK: {state['original_prompt']}\n\n"
        f"SPECIFICATION:\n{state['current_plan']}"
        + (f"\n\nPREVIOUS ATTEMPT HAD THIS ERROR - fix it:\n{prior_error}" if prior_error else "")
    )
    content = strip_markdown_fences(_gemini_generate(client, prompt, L4_SYSTEM_PROMPT))
    diff = _compute_diff(primary, old_content, content)
    return {"proposed_content": content, "proposed_diff": diff, "diff_line_count": count_diff_lines(diff)}


# AST gate & approval

def ast_gate(state: SoyGraphState) -> dict:
    """Validates Python syntax before showing the diff to the user."""
    content = state.get("proposed_content")
    if not content:
        return {
            "syntax_valid": False,
            "critique_feedback": state.get("critique_feedback") or "No content produced by Layer 4",
            "unresolved_imports": [],
            "diff_line_count": 0,
        }
    valid, error_msg = validate_python_syntax(content)
    return {
        "syntax_valid": valid,
        "critique_feedback": error_msg or None,
        "unresolved_imports": [],
        "diff_line_count": state.get("diff_line_count", 0),
    }


def handle_ast_failure(state: SoyGraphState) -> dict:
    """Increments syntax_retry_count before routing back to L4."""
    return {"syntax_retry_count": state.get("syntax_retry_count", 0) + 1}


def human_approval(state: SoyGraphState) -> dict:
    """Mandatory human review gate before writing to disk.
    Resume value: truthy = approved, falsy = rejected (routes back to L2).
    """
    approved = interrupt({"proposed_diff": state.get("proposed_diff"), "reason": "pre_exec_approval"})
    if not approved:
        return {"interrupt_reason": "rejected_by_user"}
    return {"interrupt_reason": None}


_PROTECTED_FILES = frozenset({
    "graph.py", "state.py", "config.py", "run_live.py",
    "formatters.py", "verifiers.py", "test_graph.py", "test_verifiers.py",
})


def apply_diff(state: SoyGraphState) -> dict:
    """Write approved content to disk."""
    content = state.get("proposed_content")
    target_files = state.get("target_files") or []
    if not content or not target_files:
        return {}
    primary = target_files[0]
    # Block writes to framework files in the root dir
    if os.path.basename(primary) in _PROTECTED_FILES and not os.path.dirname(primary):
        print(f"  [apply] BLOCKED: {primary!r} is a protected framework file")
        return {}
    parent = os.path.dirname(primary)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(primary, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  [apply] wrote {primary} ({len(content)} chars)")
    return {}


def interrupt_human_escalation(state: SoyGraphState) -> dict:
    """Max retries exceeded - pauses and asks the user what to do next."""
    interrupt({"reason": state.get("interrupt_reason", "circuit_breaker")})
    return {}


# Routing

def route_l1_output(state: SoyGraphState):
    """Auto mode skips the plan gate. Interactive mode pauses for human review."""
    if state.get("execution_mode", "interactive") == "auto":
        return "layer2_architect"
    return "human_plan_review"


def route_after_plan_review(state: SoyGraphState):
    if state.get("plan_approved"):
        return "layer2_architect"
    return "layer1_context_check"


def route_planning_layer(state: SoyGraphState):
    """Fans out to coworkers when fanout_depth > 0, otherwise goes straight to synthesize."""
    if state.get("fanout_depth", 0) > 0:
        base = {
            "plan_to_review": state.get("current_plan") or "",
            "system_constraints": _blackboard_field(state, "system_constraints", []),
        }
        return [Send("coworker_agent", {**base, "bias": b}) for b in ("minimalist", "cynic", "optimizer")]
    return "synthesize_plan"


def route_after_critique(state: SoyGraphState):
    if state.get("critique_iteration_count", 0) >= MAX_CRITIQUE_CYCLES:
        return "interrupt_human_escalation"
    if state.get("critique_feedback") != "PASS":
        return "handle_critique_failure"
    return "layer4_synthesize"


def route_after_ast(state: SoyGraphState):
    if not state.get("syntax_valid", False):
        if state.get("syntax_retry_count", 0) >= MAX_SYNTAX_RETRIES:
            return "interrupt_human_escalation"
        return "handle_ast_failure"
    return "human_approval"


def route_after_human_approval(state: SoyGraphState):
    if state.get("interrupt_reason") == "rejected_by_user":
        return "layer2_architect"
    return "apply_diff"


# Graph assembly

def build_graph():
    g = StateGraph(SoyGraphState)

    g.add_node("workspace_scan", workspace_scan)
    g.add_node("layer1_context_check", layer1_context_check)
    g.add_node("human_plan_review", human_plan_review)
    g.add_node("layer2_architect", layer2_architect)
    g.add_node("coworker_agent", coworker_agent)
    g.add_node("synthesize_plan", synthesize_plan)
    g.add_node("layer3_critique", layer3_critique)
    g.add_node("handle_critique_failure", handle_critique_failure)
    g.add_node("layer4_synthesize", layer4_synthesize)
    g.add_node("ast_gate", ast_gate)
    g.add_node("handle_ast_failure", handle_ast_failure)
    g.add_node("human_approval", human_approval)
    g.add_node("apply_diff", apply_diff)
    g.add_node("interrupt_human_escalation", interrupt_human_escalation)

    g.set_entry_point("workspace_scan")
    g.add_edge("workspace_scan", "layer1_context_check")
    g.add_conditional_edges("layer1_context_check", route_l1_output, ["layer2_architect", "human_plan_review"])
    g.add_conditional_edges("human_plan_review", route_after_plan_review, ["layer2_architect", "layer1_context_check"])

    g.add_conditional_edges(
        "layer2_architect",
        route_planning_layer,
        ["coworker_agent", "synthesize_plan"],
    )
    g.add_edge("coworker_agent", "synthesize_plan")
    g.add_edge("synthesize_plan", "layer3_critique")

    g.add_conditional_edges(
        "layer3_critique",
        route_after_critique,
        ["layer4_synthesize", "handle_critique_failure", "interrupt_human_escalation"],
    )
    g.add_edge("handle_critique_failure", "layer2_architect")

    g.add_edge("layer4_synthesize", "ast_gate")

    g.add_conditional_edges(
        "ast_gate",
        route_after_ast,
        ["human_approval", "handle_ast_failure", "interrupt_human_escalation"],
    )
    g.add_edge("handle_ast_failure", "layer4_synthesize")

    g.add_conditional_edges(
        "human_approval",
        route_after_human_approval,
        ["apply_diff", "layer2_architect"],
    )
    g.add_edge("apply_diff", END)
    g.add_edge("interrupt_human_escalation", END)

    return g.compile(checkpointer=MemorySaver())


graph = build_graph()
