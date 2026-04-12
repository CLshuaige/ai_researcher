"""Global state definition for LangGraph workflow."""

from typing import TypedDict, Optional
from pathlib import Path
from typing import Any

from researcher.schemas import (
    ResearchIdea,
    ExperimentalMethod,
    ExperimentResult,
    ReviewReport,
    LiteratureReview
)


class ResearchState(TypedDict):
    """Global state for research workflow

    State tracks workflow progress and structured artifacts.
    Artifacts are also persisted to files for durability.
    """
    # Input
    input_text: str
    start_node: Optional[str]
    config: Optional[dict]
    post_config: Optional[dict]
    run_mode: Optional[str]
    project_id: str
    run_id: Optional[str]

    # Research artifacts (structured objects passed between nodes)
    task: Optional[str]
    literature: Optional[LiteratureReview]
    idea: Optional[ResearchIdea]
    method: Optional[ExperimentalMethod]
    results: Optional[ExperimentResult]
    paper: Optional[str]
    referee: Optional[ReviewReport]

    # Literature review outputs
    keywords: Optional[str]
    metadata: Optional[list]

    # Workspace management
    workspace_dir: Path
    project_name: str

    # Workflow metadata
    stage: str
    error: Optional[str]
    next_node: Optional[str]

    # Session management
    session_id: str
    opencode: Optional[dict]
    cancel_event: Optional[Any]
