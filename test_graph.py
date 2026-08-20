"""
Phase 1 unit + integration tests — state machine correctness, no LLM calls.
"""

import graph as graph_module
import pytest

from langgraph.types import Command

from state import CoworkerFinding, CoworkerInput, SharedBlackboard, SoyGraphState, reduce_findings
from graph import (
    MAX_CRITIQUE_CYCLES,
    MAX_SYNTAX_RETRIES,
    build_graph,
    route_after_ast,
    route_after_critique,
    route_after_human_approval,
    route_planning_layer,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def base_state(**overrides) -> SoyGraphState:
    defaults: SoyGraphState = {
        "original_prompt": "Add a login page",
        "target_files": ["auth/login.py"],
        "current_plan": None,
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
    }
    defaults.update(overrides)
    return defaults


def _finding(bias: str) -> CoworkerFinding:
    return CoworkerFinding(
        agent_id=f"cw-{bias}",
        perspective_bias=bias,
        assessment="stub",
        suggested_modifications=[],
    )


# ── State initialisation & immutability ──────────────────────────────────────

class TestStateInit:
    def test_original_prompt_preserved(self):
        assert base_state()["original_prompt"] == "Add a login page"

    def test_original_prompt_not_mutated_by_update(self):
        state = base_state()
        merged = {**state, **{"current_plan": "some plan"}}
        assert merged["original_prompt"] == state["original_prompt"]

    def test_coworker_findings_starts_empty(self):
        assert base_state()["coworker_findings"] == []

    def test_default_loop_counters_zero(self):
        state = base_state()
        assert state["critique_iteration_count"] == 0
        assert state["syntax_retry_count"] == 0


# ── reduce_findings reducer ───────────────────────────────────────────────────

class TestReduceFindings:
    def test_nonempty_new_appends(self):
        existing = [_finding("minimalist")]
        result = reduce_findings(existing, [_finding("cynic")])
        assert len(result) == 2
        assert result[0].perspective_bias == "minimalist"
        assert result[1].perspective_bias == "cynic"

    def test_empty_new_resets_to_empty(self):
        """synthesize_plan emits [] as a reset sentinel — must wipe existing findings."""
        existing = [_finding("minimalist"), _finding("cynic")]
        assert reduce_findings(existing, []) == []

    def test_three_concurrent_sends_accumulate(self):
        """Simulate three coworkers whose results are merged sequentially by LangGraph."""
        result = []
        for bias in ["minimalist", "cynic", "optimizer"]:
            result = reduce_findings(result, [_finding(bias)])
        assert len(result) == 3
        assert {f.perspective_bias for f in result} == {"minimalist", "cynic", "optimizer"}

    def test_reset_then_accumulate(self):
        """After reset sentinel, new findings start fresh."""
        existing = [_finding("minimalist"), _finding("cynic")]
        after_reset = reduce_findings(existing, [])
        after_new = reduce_findings(after_reset, [_finding("optimizer")])
        assert len(after_new) == 1
        assert after_new[0].perspective_bias == "optimizer"


# ── CoworkerInput & fan-out routing ──────────────────────────────────────────

class TestCoworkerFanOut:
    def test_route_fanout_returns_three_sends_when_depth_gt_zero(self):
        state = base_state(fanout_depth=1, current_plan="plan v1")
        result = route_planning_layer(state)
        assert isinstance(result, list) and len(result) == 3
        assert all(s.node == "coworker_agent" for s in result)

    def test_route_no_fanout_when_depth_zero(self):
        assert route_planning_layer(base_state(fanout_depth=0)) == "synthesize_plan"

    def test_send_payloads_carry_distinct_biases(self):
        sends = route_planning_layer(base_state(fanout_depth=1, current_plan="plan"))
        assert {s.arg["bias"] for s in sends} == {"minimalist", "cynic", "optimizer"}

    def test_send_payloads_are_coworker_input_shaped(self):
        sends = route_planning_layer(base_state(fanout_depth=1, current_plan="the plan"))
        for send in sends:
            assert "bias" in send.arg
            assert "plan_to_review" in send.arg
            assert "system_constraints" in send.arg
            assert send.arg["plan_to_review"] == "the plan"


# ── Circuit breaker routing (unit) ────────────────────────────────────────────

class TestCircuitBreakers:
    def test_critique_pass_goes_to_l4(self):
        state = base_state(critique_iteration_count=0, critique_feedback="PASS")
        assert route_after_critique(state) == "layer4_synthesize"

    def test_critique_fail_under_limit_routes_to_handler(self):
        state = base_state(critique_iteration_count=0, critique_feedback="FAIL: bad plan")
        assert route_after_critique(state) == "handle_critique_failure"

    def test_critique_at_limit_escalates(self):
        assert route_after_critique(base_state(critique_iteration_count=MAX_CRITIQUE_CYCLES)) == "interrupt_human_escalation"

    def test_critique_above_limit_escalates(self):
        assert route_after_critique(base_state(critique_iteration_count=MAX_CRITIQUE_CYCLES + 1)) == "interrupt_human_escalation"

    def test_ast_pass_goes_to_human_approval(self):
        assert route_after_ast(base_state(syntax_valid=True)) == "human_approval"

    def test_ast_fail_under_retry_limit_routes_to_handler(self):
        assert route_after_ast(base_state(syntax_valid=False, syntax_retry_count=0)) == "handle_ast_failure"

    def test_ast_fail_at_retry_limit_escalates(self):
        assert route_after_ast(base_state(syntax_valid=False, syntax_retry_count=MAX_SYNTAX_RETRIES)) == "interrupt_human_escalation"

    def test_ast_fail_above_retry_limit_escalates(self):
        assert route_after_ast(base_state(syntax_valid=False, syntax_retry_count=MAX_SYNTAX_RETRIES + 1)) == "interrupt_human_escalation"


# ── human_approval reject routing ────────────────────────────────────────────

class TestHumanApproval:
    def test_approved_routes_to_apply_diff(self):
        assert route_after_human_approval(base_state(interrupt_reason=None)) == "apply_diff"

    def test_rejected_routes_to_layer2(self):
        assert route_after_human_approval(base_state(interrupt_reason="rejected_by_user")) == "layer2_architect"

    def test_other_interrupt_reason_routes_to_apply_diff(self):
        # Any non-rejected interrupt_reason falls through to apply_diff
        assert route_after_human_approval(base_state(interrupt_reason="something_else")) == "apply_diff"


# ── Integration: counter increment nodes ─────────────────────────────────────

class TestCounterNodes:
    def test_handle_critique_failure_increments(self):
        from graph import handle_critique_failure
        result = handle_critique_failure(base_state(critique_iteration_count=1))
        assert result["critique_iteration_count"] == 2

    def test_handle_ast_failure_increments(self):
        from graph import handle_ast_failure
        result = handle_ast_failure(base_state(syntax_retry_count=1))
        assert result["syntax_retry_count"] == 2

    def test_handle_critique_failure_from_zero(self):
        from graph import handle_critique_failure
        result = handle_critique_failure(base_state(critique_iteration_count=0))
        assert result["critique_iteration_count"] == 1


# ── Integration: full graph ───────────────────────────────────────────────────

class TestGraphIntegration:
    def test_graph_runs_to_human_approval_interrupt(self):
        g = build_graph()
        config = {"configurable": {"thread_id": "int-1"}}
        g.invoke(base_state(), config=config)
        assert g.get_state(config).next == ("human_approval",)

    def test_approve_reaches_end(self):
        g = build_graph()
        config = {"configurable": {"thread_id": "int-2"}}
        g.invoke(base_state(), config=config)
        g.invoke(Command(resume=True), config=config)
        assert g.get_state(config).next == ()

    def test_reject_routes_back_to_human_approval(self):
        """Command(resume=False) → rejected_by_user → L2 → ... → human_approval again."""
        g = build_graph()
        config = {"configurable": {"thread_id": "int-rej"}}
        g.invoke(base_state(), config=config)
        g.invoke(Command(resume=False), config=config)
        snap = g.get_state(config)
        assert snap.next == ("human_approval",)

    def test_reject_clears_interrupt_reason_on_re_approval(self):
        """After reject→re-plan→approve, interrupt_reason must not be 'rejected_by_user'."""
        g = build_graph()
        config = {"configurable": {"thread_id": "int-rej2"}}
        g.invoke(base_state(), config=config)
        g.invoke(Command(resume=False), config=config)  # reject → back to L2 → human_approval
        g.invoke(Command(resume=True), config=config)   # approve
        snap = g.get_state(config)
        assert snap.next == ()
        assert snap.values.get("interrupt_reason") != "rejected_by_user"

    def test_critique_loopback_increments_and_escalates(self, monkeypatch):
        """L3 always failing must escalate after MAX_CRITIQUE_CYCLES, not loop forever."""
        def failing_critique(state):
            return {"critique_feedback": "FAIL: mock structural flaw"}

        monkeypatch.setattr(graph_module, "layer3_critique", failing_critique)
        g = build_graph()
        config = {"configurable": {"thread_id": "int-3"}}
        g.invoke(base_state(), config=config)

        snap = g.get_state(config)
        assert snap.values["critique_iteration_count"] == MAX_CRITIQUE_CYCLES
        assert snap.next == ("interrupt_human_escalation",)

    def test_ast_failure_loopback_increments_and_escalates(self, monkeypatch):
        """AST always failing must escalate after MAX_SYNTAX_RETRIES, not loop forever."""
        def failing_ast(state):
            return {"syntax_valid": False, "unresolved_imports": ["missing_mod"]}

        monkeypatch.setattr(graph_module, "ast_gate", failing_ast)
        g = build_graph()
        config = {"configurable": {"thread_id": "int-4"}}
        g.invoke(base_state(), config=config)

        snap = g.get_state(config)
        assert snap.values["syntax_retry_count"] == MAX_SYNTAX_RETRIES
        assert snap.next == ("interrupt_human_escalation",)

    def test_second_cycle_findings_not_contaminated_by_first(self, monkeypatch):
        """coworker_findings are reset by synthesize_plan after each cycle — don't accumulate.
        synthesize_plan consumes and clears findings before routing onward, so both checkpoints
        at human_approval show [] rather than the 3+3=6 that operator.add without reset would give.
        """
        g = build_graph()
        config = {"configurable": {"thread_id": "int-5"}}
        g.invoke(base_state(), config=config)

        # Cycle 1: synthesize_plan already reset findings before reaching human_approval
        snap1 = g.get_state(config)
        assert snap1.next == ("human_approval",)
        assert snap1.values["coworker_findings"] == []  # consumed and reset by synthesize_plan

        # Reject → cycle 2 runs → synthesize_plan resets again
        g.invoke(Command(resume=False), config=config)
        snap2 = g.get_state(config)
        assert snap2.next == ("human_approval",)
        assert snap2.values["coworker_findings"] == []  # still reset, not 6 accumulated items

    def test_null_diff_from_l4_does_not_false_pass_ast_gate(self, monkeypatch):
        """Regression: ast_gate must not route to human_approval when proposed_diff is None.
        A failed subprocess sets proposed_diff=None; ast_gate previously false-passed because
        ast.parse('') returns True, bypassing retries and presenting a blank diff to the user.
        """
        def l4_always_fails(state):
            return {"proposed_diff": None, "critique_feedback": "claude CLI not found"}

        monkeypatch.setattr(graph_module, "layer4_synthesize", l4_always_fails)
        g = build_graph()
        config = {"configurable": {"thread_id": "null-diff"}}
        g.invoke(base_state(), config=config)

        snap = g.get_state(config)
        # Must escalate after MAX_SYNTAX_RETRIES, never reach human_approval
        assert snap.next != ("human_approval",)
        assert snap.next == ("interrupt_human_escalation",)
        assert snap.values["syntax_retry_count"] == MAX_SYNTAX_RETRIES

    def test_original_prompt_preserved_through_full_graph(self):
        g = build_graph()
        config = {"configurable": {"thread_id": "int-6"}}
        prompt = "Add a login page"
        g.invoke(base_state(original_prompt=prompt), config=config)
        g.invoke(Command(resume=True), config=config)
        assert g.get_state(config).values["original_prompt"] == prompt
