"""
SoyLangGraph — Phase 1 skeleton.
Mock nodes wire the main spine; real LLM calls slot in later.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send, interrupt

from state import CoworkerFinding, CoworkerInput, SharedBlackboard, SoyGraphState

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


# ── Mock spine nodes ──────────────────────────────────────────────────────────

def layer1_context_check(state: SoyGraphState) -> dict:
    """Gemini Flash context analysis — mock returns a minimal plan."""
    return {"current_plan": f"[L1 stub] analyse: {state['original_prompt'][:80]}"}


def layer2_architect(state: SoyGraphState) -> dict:
    """Gemini Flash system architect — sets fanout_depth."""
    return {
        "current_plan": "[L2 stub] architecture plan",
        "fanout_depth": 1,
    }


def coworker_agent(state: CoworkerInput) -> dict:
    """Mock coworker sub-agent; receives CoworkerInput-shaped state from Send().
    IMPORTANT: never return coworker_findings=[] — omit the key if no findings.
    An empty list is the reset sentinel in reduce_findings and will wipe sibling results.
    """
    bias = state.get("bias", "unknown")
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


def synthesize_plan(state: SoyGraphState) -> dict:
    """Distill coworker findings into current_plan, then reset the buffer."""
    biases = [f.perspective_bias for f in state.get("coworker_findings", [])]
    return {
        "current_plan": f"[L2 consensus] biases={biases}",
        "coworker_findings": [],  # reset sentinel — reduce_findings treats [] as replace
    }


def layer3_critique(state: SoyGraphState) -> dict:
    """Antigravity Flash + CLI — mock always passes. Returns 'PASS' or failure text."""
    return {"critique_feedback": "PASS"}


def handle_critique_failure(state: SoyGraphState) -> dict:
    """Increments critique_iteration_count before routing back to L2."""
    return {"critique_iteration_count": state.get("critique_iteration_count", 0) + 1}


def layer4_synthesize(state: SoyGraphState) -> dict:
    """Claude Code headless subprocess — mock returns a stub diff."""
    return {
        "proposed_diff": "--- a/stub.py\n+++ b/stub.py\n@@ -0,0 +1 @@\n+# stub\n",
        "diff_line_count": 4,
    }


def ast_gate(state: SoyGraphState) -> dict:
    """Deterministic AST/syntax check — mock always passes."""
    return {"syntax_valid": True, "unresolved_imports": []}


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
