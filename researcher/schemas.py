from typing import Annotated, List, Optional, Dict, Any
from datetime import datetime
import json

from pydantic import BaseModel, Field, BeforeValidator
from .latex.utils import extract_latex_code


# Prevent the model from producing incorrect outputs at the expected string positions.
def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        parts = [_normalize_string(item) for item in value]
        parts = [part for part in parts if part]
        return "\n".join(parts)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value).strip()
    return str(value).strip()


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items: List[str] = []
        for item in value:
            if isinstance(item, (list, tuple, set)):
                items.extend(_normalize_string_list(item))
            else:
                normalized = _normalize_string(item)
                if normalized:
                    items.append(normalized)
        return items
    normalized = _normalize_string(value)
    return [normalized] if normalized else []


def _normalize_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except Exception:
            decoded = [segment.strip() for segment in text.split(",")]
        return _normalize_int_list(decoded)
    if not isinstance(value, (list, tuple, set)):
        value = [value]

    numbers: List[int] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, bool):
            numbers.append(int(item))
            continue
        if isinstance(item, (int, float)):
            numbers.append(int(item))
            continue
        text = str(item).strip()
        if not text:
            continue
        try:
            numbers.append(int(float(text)))
        except Exception:
            continue
    return numbers


NormalizedStr = Annotated[str, BeforeValidator(_normalize_string)]
NormalizedStrList = Annotated[List[str], BeforeValidator(_normalize_string_list)]
NormalizedIntList = Annotated[List[int], BeforeValidator(_normalize_int_list)]


class IdeaCandidate(BaseModel):
    """Single research idea candidate"""
    content: NormalizedStr = Field(description="Idea description")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Idea quality score")
    strengths: NormalizedStrList = Field(default_factory=list, description="Idea strengths")
    weaknesses: NormalizedStrList = Field(default_factory=list, description="Idea weaknesses")

    basis: NormalizedStr = Field(description="Key scientific basis")
    components: NormalizedStrList = Field(default_factory=list, description="Implementation components")

    round: int = Field(default=0, description="Round when proposed")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Timestamp in ISO format")


class ResearchIdea(BaseModel):
    """Research hypothesis with multiple candidates"""
    candidates: List[IdeaCandidate] = Field(default_factory=list)
    selected_index: int = Field(default=0, description="Index of selected idea")
    debate_rounds: int = Field(default=0)

    @property
    def selected_idea(self) -> Optional[IdeaCandidate]:
        """Get the selected idea"""
        if not self.candidates:
            return None
        return self.candidates[self.selected_index]

    def add_candidate(self, content: str, round: int, score: float = 0.0) -> None:
        """Add new idea candidate"""
        candidate = IdeaCandidate(content=content, score=score, round=round)
        self.candidates.append(candidate)

    def add_criticism(self, candidate_index: int, criticism: str) -> None:
        """Add criticism to specific candidate"""
        if 0 <= candidate_index < len(self.candidates):
            self.candidates[candidate_index].weaknesses.append(criticism)

    def rank_candidates(self) -> None:
        """Sort candidates by score and select best"""
        self.candidates.sort(key=lambda x: x.score, reverse=True)
        self.selected_index = 0

    def to_markdown(self) -> str:
        """Export to markdown format"""
        lines = ["# Research Ideas\n"]
        lines.append(f"Debate Rounds: {self.debate_rounds}\n")
        lines.append(f"Total Candidates: {len(self.candidates)}\n\n")

        for i, candidate in enumerate(self.candidates):
            marker = "**[SELECTED]**" if i == self.selected_index else ""
            lines.append(f"## Idea {i+1} {marker}\n")
            lines.append(f"Score: {candidate.score:.2f}\n")
            lines.append(f"Round: {candidate.round}\n\n")
            lines.append(f"{candidate.content}\n\n")

            if i == self.selected_index:
                if candidate.basis:
                    lines.append(f"### Basis\n{candidate.basis}\n")
                if candidate.components:
                    lines.append("### Components\n")
                    for j, component in enumerate(candidate.components):
                        lines.append(f"{j+1}. {component}\n")
                    lines.append("\n")

            if candidate.strengths:
                lines.append("### Strengths\n")
                for j, strength in enumerate(candidate.strengths, 1):
                    lines.append(f"{j}. {strength}\n")
                lines.append("\n")

            if candidate.weaknesses:
                lines.append("### Weaknesses\n")
                for j, weakness in enumerate(candidate.weaknesses, 1):
                    lines.append(f"{j}. {weakness}\n")
                lines.append("\n")

        return "".join(lines)


class MethodStep(BaseModel):
    """Single experimental step with role assignment"""
    step_id: int = Field(description="Step ID (1-indexed)")
    description: NormalizedStr = Field(description="Step description")
    assignee: NormalizedStr = Field(description="RA or Engineer")
    dependencies: NormalizedIntList = Field(default_factory=list, description="Dependent step IDs")
    expected_output: NormalizedStr = Field(default="", description="Expected output description")
    #implement_guidance: str = Field(default="", description="Implementation guidance")


class TaskAssignment(BaseModel):
    """Task assignment for RA or Engineer"""
    role: NormalizedStr = Field(description="RA or Engineer")
    tasks: NormalizedStrList = Field(default_factory=list)
    dependencies: NormalizedStrList = Field(default_factory=list)


class ExperimentalMethod(BaseModel):
    """Experimental method design"""
    overview: NormalizedStr = Field(description="Method overview")
    steps: List[MethodStep] = Field(default_factory=list, description="Structured execution steps")
    execution_order: NormalizedIntList = Field(default_factory=list, description="Step execution order")
    assignments: List[TaskAssignment] = Field(default_factory=list)
    resources: Dict[str, Any] = Field(default_factory=dict, description="Required resources")
    debate_rounds: int = Field(default=0)
    criticisms: NormalizedStrList = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Export to markdown format"""
        lines = ["# Experimental Method\n\n"]
        lines.append(f"## Overview\n{self.overview}\n\n")

        if self.steps:
            lines.append("## Execution Steps\n")
            for step in self.steps:
                lines.append(f"### Step {step.step_id}: {step.description}\n")
                lines.append(f"- **Assignee**: {step.assignee}\n")
                if step.dependencies:
                    lines.append(f"- **Dependencies**: {', '.join(map(str, step.dependencies))}\n")
                # if step.implement_guidance:
                #     lines.append(f"- **Implement Guidance**: {step.implement_guidance}\n")
                if step.expected_output:
                    lines.append(f"- **Expected Output**: {step.expected_output}\n")
                lines.append("\n")

        if self.execution_order:
            lines.append(f"## Execution Order\n{' → '.join(map(str, self.execution_order))}\n\n")

        if self.assignments:
            lines.append("## Task Assignments\n")
            for assignment in self.assignments:
                lines.append(f"### {assignment.role}\n")
                for task in assignment.tasks:
                    lines.append(f"- {task}\n")
                if assignment.dependencies:
                    lines.append(f"Dependencies: {', '.join(assignment.dependencies)}\n")
                lines.append("\n")

        if self.resources:
            lines.append("## Resources\n")
            for key, value in self.resources.items():
                lines.append(f"- {key}: {value}\n")
            lines.append("\n")

        return "".join(lines)


class ExperimentResult(BaseModel):
    """Experimental results"""
    summary: NormalizedStr = Field(description="Result summary")
    data_paths: NormalizedStrList = Field(default_factory=list)
    figure_paths: NormalizedStrList = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    analysis: NormalizedStr = Field(default="")

    def to_markdown(self) -> str:
        """Export to markdown format"""
        lines = ["# Experimental Results\n\n"]
        lines.append(f"## Summary\n{self.summary}\n\n")

        if self.metrics:
            lines.append("## Metrics\n")
            for key, value in self.metrics.items():
                lines.append(f"- {key}: {value}\n")
            lines.append("\n")

        if self.figure_paths:
            lines.append("## Figures\n")
            for path in self.figure_paths:
                lines.append(f"![Figure]({path})\n")
            lines.append("\n")

        if self.data_paths:
            lines.append("## Data Files\n")
            max_display = 20
            if len(self.data_paths) <= max_display:
                display_paths = self.data_paths
            else:
                display_paths = self.data_paths[:max_display-1]
                truncation_message = f"... and {len(self.data_paths) - (max_display - 1)} more files"
                display_paths.append(truncation_message)

            for path in display_paths:
                lines.append(f"- {path}\n")
            lines.append("\n")

        if self.analysis:
            lines.append(f"## Analysis\n{self.analysis}\n\n")

        return "".join(lines)


class ReviewReport(BaseModel):
    """ICML-style review report"""
    summary: NormalizedStr = Field(description="Review summary")
    strengths: NormalizedStrList = Field(default_factory=list)
    weaknesses: NormalizedStrList = Field(default_factory=list)
    questions: NormalizedStrList = Field(default_factory=list)
    score: int = Field(ge=1, le=10, description="Overall score (1-10)")
    confidence: int = Field(ge=1, le=5, description="Reviewer confidence (1-5)")
    recommendation: NormalizedStr = Field(description="Accept/Reject/Revise")

    def to_markdown(self) -> str:
        """Export to markdown format (ICML style)"""
        lines = ["# Review Report\n\n"]
        lines.append(f"**Score**: {self.score}/10\n")
        lines.append(f"**Confidence**: {self.confidence}/5\n")
        lines.append(f"**Recommendation**: {self.recommendation}\n\n")

        lines.append(f"## Summary\n{self.summary}\n\n")

        if self.strengths:
            lines.append("## Strengths\n")
            for i, strength in enumerate(self.strengths, 1):
                lines.append(f"{i}. {strength}\n")
            lines.append("\n")

        if self.weaknesses:
            lines.append("## Weaknesses\n")
            for i, weakness in enumerate(self.weaknesses, 1):
                lines.append(f"{i}. {weakness}\n")
            lines.append("\n")

        if self.questions:
            lines.append("## Questions\n")
            for i, question in enumerate(self.questions, 1):
                lines.append(f"{i}. {question}\n")
            lines.append("\n")

        return "".join(lines)


class LiteratureItem(BaseModel):
    """Single literature item"""
    title: NormalizedStr
    authors: NormalizedStrList = Field(default_factory=list)
    abstract: NormalizedStr = Field(default="")
    url: Optional[str] = None
    year: Optional[int] = None


class LiteratureReview(BaseModel):
    """Literature review collection"""
    items: List[LiteratureItem] = Field(default_factory=list)
    synthesis: NormalizedStr = Field(default="", description="Synthesized review")

    def to_markdown(self) -> str:
        """Export to markdown format"""
        lines = ["# Literature Review\n\n"]

        if self.synthesis:
            lines.append(f"## Synthesis\n{self.synthesis}\n\n")

        if self.items:
            lines.append("## References\n\n")
            for i, item in enumerate(self.items, 1):
                lines.append(f"### {i}. {item.title}\n")
                if item.authors:
                    lines.append(f"**Authors**: {', '.join(item.authors)}\n")
                if item.year:
                    lines.append(f"**Year**: {item.year}\n")
                if item.url:
                    lines.append(f"**URL**: {item.url}\n")
                lines.append(f"\n{item.abstract}\n\n")

        return "".join(lines)
    
    def to_latex(self) -> str:
        return extract_latex_code(self.synthesis)

class ChatResult(BaseModel):
    """Chat result for group chat"""
    chat_history: List[Dict[str, Any]]
