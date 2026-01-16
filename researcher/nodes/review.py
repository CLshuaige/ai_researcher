from typing import Dict, Any

from autogen import UserProxyAgent

from researcher.state import ResearchState
from researcher.schemas import ReviewReport
from researcher.agents import ReviewerAgent
from researcher.utils import (
    save_markdown,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
)
from researcher.prompts.templates import REVIEW_PROMPT
from researcher.exceptions import WorkflowError


def review_node(state: ResearchState) -> Dict[str, Any]:
    """Review research paper"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "review", "Starting review")

    try:
        paper = load_artifact_from_file(workspace_dir, "paper")
        if not paper:
            # Try .md extension
            paper_md_path = get_artifact_path(workspace_dir, "paper").with_suffix('.md')
            if paper_md_path.exists():
                with open(paper_md_path, 'r', encoding='utf-8') as f:
                    paper = f.read()

        if not paper:
            raise WorkflowError("Paper file not found")

        llm_config = get_llm_config()

        prompt = REVIEW_PROMPT.format(paper=paper)

        reviewer = ReviewerAgent().create_assistant(llm_config)
        user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=0)

        log_stage(workspace_dir, "review", "Generating review")
        user_proxy.initiate_chat(reviewer, message=prompt)

        review_text = user_proxy.last_message()["content"]

        referee = _parse_review(review_text)

        referee_path = get_artifact_path(workspace_dir, "referee")
        save_markdown(referee.to_markdown(), referee_path)

        log_stage(workspace_dir, "review", f"Completed. Score: {referee.score}/10, Recommendation: {referee.recommendation}")

        return {
            "task": load_artifact_from_file(workspace_dir, "task"),
            "paper": paper,
            "referee": referee,
            "stage": "review"
        }

    except Exception as e:
        log_stage(workspace_dir, "review", f"Error: {str(e)}")
        raise WorkflowError(f"Review failed: {str(e)}")


def _parse_review(review_text: str) -> ReviewReport:
    """Parse review text into ReviewReport object"""
    lines = review_text.split('\n')

    summary = ""
    strengths = []
    weaknesses = []
    questions = []
    score = 5
    confidence = 3
    recommendation = "Borderline"

    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "summary" in line.lower() and line.endswith(':'):
            current_section = "summary"
        elif "strength" in line.lower() and line.endswith(':'):
            current_section = "strengths"
        elif "weakness" in line.lower() and line.endswith(':'):
            current_section = "weaknesses"
        elif "question" in line.lower() and line.endswith(':'):
            current_section = "questions"
        elif "score" in line.lower() and ':' in line:
            try:
                score = int(line.split(':')[1].strip().split('/')[0])
            except:
                pass
        elif "confidence" in line.lower() and ':' in line:
            try:
                confidence = int(line.split(':')[1].strip().split('/')[0])
            except:
                pass
        elif "recommendation" in line.lower() and ':' in line:
            recommendation = line.split(':')[1].strip()
        elif current_section == "summary" and not line.startswith('#'):
            summary += line + " "
        elif current_section == "strengths" and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
            strengths.append(line.lstrip('-•0123456789. '))
        elif current_section == "weaknesses" and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
            weaknesses.append(line.lstrip('-•0123456789. '))
        elif current_section == "questions" and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
            questions.append(line.lstrip('-•0123456789. '))

    return ReviewReport(
        summary=summary.strip(),
        strengths=strengths,
        weaknesses=weaknesses,
        questions=questions,
        score=score,
        confidence=confidence,
        recommendation=recommendation
    )
