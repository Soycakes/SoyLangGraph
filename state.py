from dataclasses import dataclass, field
from typing import Annotated, List, Literal, Optional, TypedDict


@dataclass
class CoworkerFinding:
    agent_id: str
    perspective_bias: str
    assessment: str
    suggested_modifications: List[str]


@dataclass
class SharedBlackboard:
    system_constraints: List[str] = field(default_factory=list)


class CoworkerInput(TypedDict):
    bias: Literal["minimalist", "cynic", "optimizer"]
    plan_to_review: str
    system_constraints: List[str]


def reduce_findings(
    existing: List[CoworkerFinding], new: List[CoworkerFinding]
) -> List[CoworkerFinding]:
    """Reducer for coworker_findings. Empty list resets, non-empty appends."""
    if not new:
        return []
    return existing + new


class SoyGraphState(TypedDict):
    # Set at start, never changed
    original_prompt: str
    target_files: List[str]

    # Layer artifacts
    current_plan: Optional[str]
    layer_blackboard: SharedBlackboard

    # Parallel fanout: reduce_findings accumulates results, synthesize_plan resets with []
    coworker_findings: Annotated[List[CoworkerFinding], reduce_findings]

    # Code generation artifacts
    proposed_content: Optional[str]  # full file content from L4
    proposed_diff: Optional[str]     # computed for display only
    critique_feedback: Optional[str]

    # Quality flags
    syntax_valid: bool
    unresolved_imports: List[str]
    diff_line_count: int

    # Loop controls
    critique_iteration_count: int  # escalates after MAX_CRITIQUE_CYCLES
    syntax_retry_count: int        # escalates after MAX_SYNTAX_RETRIES
    fanout_depth: int
    interrupt_reason: Optional[str]

    # Workspace & plan gate
    execution_mode: str             # "interactive" or "auto"
    workspace_context: Optional[dict]
    plan_feedback: Optional[str]    # user correction when plan is rejected
    plan_approved: Optional[bool]
