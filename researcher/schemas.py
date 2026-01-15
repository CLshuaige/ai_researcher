from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class IdeaCandidate(BaseModel):
    """Single research idea candidate"""
    content: str = Field(description="Idea description")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Idea quality score")
    criticisms: List[str] = Field(default_factory=list, description="Criticism history")
    round: int = Field(default=0, description="Round when proposed")
    timestamp: datetime = Field(default_factory=datetime.now)


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
            self.candidates[candidate_index].criticisms.append(criticism)

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

            if candidate.criticisms:
                lines.append("### Criticisms\n")
                for j, crit in enumerate(candidate.criticisms, 1):
                    lines.append(f"{j}. {crit}\n")
                lines.append("\n")

        return "".join(lines)


class TaskAssignment(BaseModel):
    """Task assignment for RA or Engineer"""
    role: str = Field(description="RA or Engineer")
    tasks: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)


class ExperimentalMethod(BaseModel):
    """Experimental method design"""
    overview: str = Field(description="Method overview")
    steps: List[str] = Field(default_factory=list)
    assignments: List[TaskAssignment] = Field(default_factory=list)
    resources: Dict[str, Any] = Field(default_factory=dict, description="Required resources")
    debate_rounds: int = Field(default=0)
    criticisms: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Export to markdown format"""
        lines = ["# Experimental Method\n\n"]
        lines.append(f"## Overview\n{self.overview}\n\n")

        if self.steps:
            lines.append("## Steps\n")
            for i, step in enumerate(self.steps, 1):
                lines.append(f"{i}. {step}\n")
            lines.append("\n")

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
    summary: str = Field(description="Result summary")
    data_paths: List[str] = Field(default_factory=list)
    figure_paths: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    analysis: str = Field(default="")

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
            for path in self.data_paths:
                lines.append(f"- {path}\n")
            lines.append("\n")

        if self.analysis:
            lines.append(f"## Analysis\n{self.analysis}\n\n")

        return "".join(lines)


class ReviewReport(BaseModel):
    """ICML-style review report"""
    summary: str = Field(description="Review summary")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    score: int = Field(ge=1, le=10, description="Overall score (1-10)")
    confidence: int = Field(ge=1, le=5, description="Reviewer confidence (1-5)")
    recommendation: str = Field(description="Accept/Reject/Revise")

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
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str = Field(default="")
    url: Optional[str] = None
    year: Optional[int] = None


class LiteratureReview(BaseModel):
    """Literature review collection"""
    items: List[LiteratureItem] = Field(default_factory=list)
    synthesis: str = Field(default="", description="Synthesized review")

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
