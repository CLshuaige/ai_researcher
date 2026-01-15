from typing import TypedDict, Optional, List, Dict, Any
from pathlib import Path

from researcher.schemas import (
    ResearchIdea,
    ExperimentalMethod,
    ExperimentResult,
    ReviewReport,
    LiteratureReview
)


class ResearchState(TypedDict):
    """Global state for research workflow"""
    # Input and task definition
    input_text: str
    task: Optional[str]

    # Research artifacts (structured)
    literature: Optional[LiteratureReview]
    idea: Optional[ResearchIdea]
    method: Optional[ExperimentalMethod]
    results: Optional[ExperimentResult]
    paper: Optional[str]
    referee: Optional[ReviewReport]

    # Debate tracking
    current_round: int
    debate_history: List[Dict[str, Any]]

    # Workspace management
    workspace_dir: Path
    project_name: str

    # Metadata
    stage: str
    error: Optional[str]
