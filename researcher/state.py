"""Global state definition for LangGraph workflow."""

from typing import TypedDict, Optional
from pathlib import Path

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
    task: Optional[str]
    config: Optional[dict]

    # Research artifacts (structured objects passed between nodes)
    literature: Optional[LiteratureReview]
    idea: Optional[ResearchIdea]
    method: Optional[ExperimentalMethod]
    results: Optional[ExperimentResult]
    paper: Optional[str]
    referee: Optional[ReviewReport]

    # Workspace management
    workspace_dir: Path
    project_name: str

    # Workflow metadata
    stage: str
    error: Optional[str]

    # Session management
    session_id: str
