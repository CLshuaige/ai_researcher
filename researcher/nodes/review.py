from typing import Dict, Any

from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import RevertToUserTarget

from researcher.state import ResearchState
from researcher.schemas import ReviewReport
from researcher.agents import ReviewerAgent
from researcher.utils import (
    save_markdown,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    save_agent_history,
)
from researcher.prompts.templates import REVIEW_PROMPT
from researcher.exceptions import WorkflowError


def review_node(state: ResearchState) -> Dict[str, Any]:
    """Review research report"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "review", "Starting review")

    try:
        paper = load_artifact_from_file(workspace_dir, "paper")
        if not paper:
            # Try .md extension
            paper_md_path = get_artifact_path(workspace_dir, "paper").with_suffix('.md')
            log_stage(workspace_dir, "review", f"Loading paper from markdown file: {paper_md_path}")
            if paper_md_path.exists():
                with open(paper_md_path, 'r', encoding='utf-8') as f:
                    paper = f.read()

        if not paper:
            raise WorkflowError("Paper file not found")

        llm_config = get_llm_config()

        reviewer = ReviewerAgent().create_agent(llm_config)

        pattern = DefaultPattern(
            initial_agent=reviewer,
            agents=[reviewer],
            group_manager_args={"llm_config": llm_config}
        )

        reviewer.handoffs.set_after_work(RevertToUserTarget())

        prompt = REVIEW_PROMPT.format(paper=paper)

        result, context, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=prompt,
            max_rounds=1
        )

        # Extract review from reviewer
        review_text = None
        for msg in reversed(result.chat_history):
            if msg.get("name") == reviewer.name and msg.get("content"):
                review_text = msg["content"]
                break

        if not review_text:
            raise WorkflowError("Reviewer did not generate review")

        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="review",
            messages=result.chat_history,
            agent_chat_messages=reviewer.chat_messages
        )

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
