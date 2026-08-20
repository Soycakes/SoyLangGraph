import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class CoworkerFinding(BaseModel):
    agent_id: str
    perspective_bias: str
    assessment: str
    suggested_modifications: List[str]


class SharedBlackboard(BaseModel):
    verified_files: List[str] = Field(default_factory=list)
    system_constraints: List[str] = Field(default_factory=list)
    rejected_paths: List[str] = Field(default_factory=list)
    active_revelations: List[str] = Field(default_factory=list)


class CoworkerInput(TypedDict):
    bias: Literal["minimalist", "cynic", "optimizer"]
    plan_to_review: str
    system_constraints: List[str]


def reduce_findings(
    existing: List[CoworkerFinding], new: List[CoworkerFinding]
) -> List[CoworkerFinding]:
    """Custom reducer for coworker_findings.
    Empty new list is the reset sentinel emitted by synthesize_plan after distillation.
    Non-empty list appends (accumulates concurrent Send() returns within a cycle).
    """
    if not new:
        return []
    return existing + new


class SoyGraphState(TypedDict):
    # Immutable anchor — never modified by agents
    original_prompt: str
    target_files: List[str]

    # Layer artifacts
    current_plan: Optional[str]
    layer_blackboard: SharedBlackboard

    # Parallel fan-out — reduce_findings accumulates concurrent Send() returns;
    # synthesize_plan resets the buffer each cycle by emitting [] as the sentinel.
    coworker_findings: Annotated[List[CoworkerFinding], reduce_findings]

    # Code & review artifacts
    proposed_diff: Optional[str]
    critique_feedback: Optional[str]

    # Quality flags
    syntax_valid: bool
    unresolved_imports: List[str]
    redteam_fatal_flaws: List[str]
    diff_line_count: int

    # Loop controls
    critique_iteration_count: int  # capped by max_critique_cycles
    syntax_retry_count: int        # capped at 2, then interrupt_human_escalation
    fanout_depth: int
    interrupt_reason: Optional[str]
