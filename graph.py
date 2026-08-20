"""
SoyLangGraph — orchestration graph.
Nodes with real LLM calls fall back to stubs when GEMINI_API_KEY is unset,
keeping unit tests fully isolated without any changes to the test suite.
"""

import platform
import re
import subprocess

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send, interrupt

from config import (
    COWORKER_PROMPTS,
    GEMINI_MODEL,
    L1_SYSTEM_PROMPT,
    get_gemini_client,
)
from state import CoworkerFinding, CoworkerInput, SharedBlackboard, SoyGraphState
from verifiers import count_diff_lines, extract_diff_additions, strip_markdown_fences, validate_python_syntax

# ── Config ────────────────────────────────────────────────────────────────────

MAX_CRITIQUE_CYCLES = 3
MAX_SYNTAX_RETRIES = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blackboard_field(state: SoyGraphState, field: str, default):
    """Safe access for SharedBlackboard fields — works with Pydantic models and plain dicts.
    ponytail: MemorySaver keeps Pydantic objects intact; postgres/sqlite checkpointers deserialize
    to dicts. This helper bridges both. Remove when checkpointer is finalized.
    """
    bb = state["layer_blackboard"]
    if isinstance(bb, dict):
        return bb.get(field, default)
    return getattr(bb, field, default)


def _gemini_generate(client, prompt: str, system: str) -> str:
    """Single-call wrapper around client.models.generate_content."""
    from google.genai import types
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return (response.text or "").strip()


def _extract_bullets(text: str) -> list[str]:
    return [
        line.lstrip("-•* ").strip()
        for line in text.splitlines()
        if line.strip().startswith(("-", "•", "*")) and line.strip().lstrip("-•* ")
    ]


def _parse_l1_response(text: str) -> dict:
    """Extract PLAN from L1 response. Regex handles markdown bolding Gemini sometimes adds."""
    match = re.search(r"(?:\*\*)?PLAN:?(?:\*\*)?\s*(.*?)(?=\nFILES:|\nCOMPLEXITY:|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return {"current_plan": match.group(1).strip()}
    return {"current_plan": text}  # fallback: treat whole response as plan


def _parse_coworker_response(text: str, bias: str) -> dict:
    """Parse coworker LLM output into a CoworkerFinding, or return {} on NO_ISSUES.
    Returning {} (omitting coworker_findings) is the correct reducer contract —
    an empty list [] would wipe sibling coworker results.
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


# ── Stub / pass-through architecture nodes ───────────────────────────────────

def l1_ponytail(state: SoyGraphState) -> dict:
    """Ponytail YAGNI scope filter — stub: pass-through until Gemini integration."""
    return {}


def l3_ponytail(state: SoyGraphState) -> dict:
    """Ponytail diff minimizer — stub: pass-through until diff budget logic lands."""
    return {}


def peer_verification(state: SoyGraphState) -> dict:
    """Antigravity Flash peer check for Eureka revelations — stub: always debunks."""
    return {"interrupt_reason": None}
    # Phase 1: node registered but not yet wired into Gate3 conditional routing


# ── Layer 1: Context Analyst ──────────────────────────────────────────────────

def layer1_context_check(state: SoyGraphState) -> dict:
    """Gemini Flash context analysis. Falls back to stub when GEMINI_API_KEY is unset."""
    client = get_gemini_client()
    if client is None:
        return {"current_plan": f"[L1 stub] analyse: {state['original_prompt'][:80]}"}
    text = _gemini_generate(client, state["original_prompt"], L1_SYSTEM_PROMPT)
    return _parse_l1_response(text)


# ── Layer 2: Architect + Coworker Fan-Out ─────────────────────────────────────

def layer2_architect(state: SoyGraphState) -> dict:
    """Gemini Flash system architect — stub sets fanout_depth=1. Demock in Phase 4."""
    return {
        "current_plan": state.get("current_plan") or "[L2 stub] architecture plan",
        "fanout_depth": 1,
    }


def coworker_agent(state: CoworkerInput) -> dict:
    """Gemini Flash coworker with perspective bias. Falls back to stub when GEMINI_API_KEY is unset.
    IMPORTANT: return {} (omit key) when no findings — never return coworker_findings=[].
    An empty list is the reset sentinel in reduce_findings and will wipe sibling results.
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
    """Distill coworker findings into current_plan, then reset the buffer."""
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
        "coworker_findings": [],  # reset sentinel — reduce_findings treats [] as replace
    }


# ── Layer 3: Technical Critique ───────────────────────────────────────────────

def layer3_critique(state: SoyGraphState) -> dict:
    """Antigravity Flash + CLI — stub always passes. Demock in Phase 4."""
    return {"critique_feedback": "PASS"}


def handle_critique_failure(state: SoyGraphState) -> dict:
    """Increments critique_iteration_count before routing back to L2."""
    return {"critique_iteration_count": state.get("critique_iteration_count", 0) + 1}


# ── Layer 4: Lead Synthesizer ─────────────────────────────────────────────────

def layer4_synthesize(state: SoyGraphState) -> dict:
    """Claude Code headless subprocess. Falls back to stub diff when GEMINI_API_KEY is unset."""
    client = get_gemini_client()
    if client is None:
        return {
            "proposed_diff": "--- a/stub.py\n+++ b/stub.py\n@@ -0,0 +1 @@\n+# stub\n",
            "diff_line_count": 4,
        }
    prompt = (
        f"Output ONLY a valid git unified diff for target files "
        f"{state['target_files']}. No prose or explanations.\n\n"
        f"Plan:\n{state['current_plan']}"
    )
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
            shell=(platform.system() == "Windows"),  # claude installs as .cmd on Windows
        )
    except FileNotFoundError:
        return {"proposed_diff": None, "critique_feedback": "claude CLI not found — install @anthropic-ai/claude-code"}
    except subprocess.TimeoutExpired:
        return {"proposed_diff": None, "critique_feedback": "claude subprocess timed out after 120s"}

    if result.returncode != 0:
        stderr = (result.stderr or "")[:300]
        return {"proposed_diff": None, "critique_feedback": f"claude exit {result.returncode}: {stderr}"}

    diff = result.stdout.strip()
    return {
        "proposed_diff": diff,
        "diff_line_count": count_diff_lines(strip_markdown_fences(diff)),
    }


# ── AST gate & approval ───────────────────────────────────────────────────────

def ast_gate(state: SoyGraphState) -> dict:
    """Deterministic AST/syntax check on proposed diff additions."""
    raw_diff = state.get("proposed_diff")
    if not raw_diff:
        # L4 produced no diff (subprocess error, timeout, etc.) — preserve L4's error message
        return {
            "syntax_valid": False,
            "critique_feedback": state.get("critique_feedback") or "No diff produced by Layer 4",
            "unresolved_imports": [],
            "diff_line_count": 0,
        }
    diff = strip_markdown_fences(raw_diff)
    additions = extract_diff_additions(diff)
    valid, error_msg = validate_python_syntax(additions)
    return {
        "syntax_valid": valid,
        "critique_feedback": error_msg or None,  # README §6: "Append Error to Feedback" → L4 retry context
        "unresolved_imports": [],
        "diff_line_count": count_diff_lines(diff),
    }


def handle_ast_failure(state: SoyGraphState) -> dict:
    """Increments syntax_retry_count before routing back to L4."""
    return {"syntax_retry_count": state.get("syntax_retry_count", 0) + 1}


def human_approval(state: SoyGraphState) -> dict:
    """Mandatory pre-exec human review gate.
    interrupt() pauses here with diff payload.
    Resume value: truthy = approved, falsy = rejected (routes back to L2).
    """
    approved = interrupt({"proposed_diff": state.get("proposed_diff"), "reason": "pre_exec_approval"})
    if not approved:
        return {"interrupt_reason": "rejected_by_user"}
    return {"interrupt_reason": None}


def apply_diff(state: SoyGraphState) -> dict:
    """Write approved diff to disk — stub logs intent only."""
    return {}


def interrupt_human_escalation(state: SoyGraphState) -> dict:
    """Hard-stop: circuit breaker tripped, needs human direction."""
    interrupt({"reason": state.get("interrupt_reason", "circuit_breaker")})
    return {}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_l1_output(state: SoyGraphState):
    """Conditional Ponytail scope prune — stub always bypasses."""
    # Real impl: if scope_creep_detected → "l1_ponytail"
    return "layer2_architect"


def route_planning_layer(state: SoyGraphState):
    """Fan-out to coworkers when fanout_depth > 0, else go straight to synthesize."""
    if state.get("fanout_depth", 0) > 0:
        payload = CoworkerInput(
            plan_to_review=state.get("current_plan") or "",
            system_constraints=_blackboard_field(state, "system_constraints", []),
            bias="minimalist",  # overridden per-Send below
        )
        return [
            Send("coworker_agent", {**payload, "bias": "minimalist"}),
            Send("coworker_agent", {**payload, "bias": "cynic"}),
            Send("coworker_agent", {**payload, "bias": "optimizer"}),
        ]
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


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(SoyGraphState)

    # Spine nodes
    g.add_node("layer1_context_check", layer1_context_check)
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
    # Architecture stub nodes (topology matches README Section 6 flowchart)
    g.add_node("l1_ponytail", l1_ponytail)
    g.add_node("l3_ponytail", l3_ponytail)
    # Phase 1: peer_verification registered but not yet wired into Gate3 conditional routing
    g.add_node("peer_verification", peer_verification)

    g.set_entry_point("layer1_context_check")
    g.add_conditional_edges("layer1_context_check", route_l1_output, ["l1_ponytail", "layer2_architect"])
    g.add_edge("l1_ponytail", "layer2_architect")

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

    # ponytail: l3_ponytail not yet wired into a conditional route; add when diff budget logic lands
    g.add_edge("l3_ponytail", "layer4_synthesize")

    return g.compile(checkpointer=MemorySaver())


graph = build_graph()
