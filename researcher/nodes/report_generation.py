from typing import Dict, Any

from researcher.state import ResearchState
from researcher.agents import WriterAgent
from researcher.config import config
from researcher.utils import save_markdown, log_stage, get_artifact_path
from researcher.prompts.templates import PAPER_WRITING_PROMPT
from researcher.llm import get_llm_client
from researcher.exceptions import WorkflowError


def report_generation_node(state: ResearchState) -> Dict[str, Any]:
    """Generate research report"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "report_generation", "Starting report generation")

    try:
        writer = WriterAgent()
        llm_client = get_llm_client(config.model)

        idea_content = state["idea"].selected_idea.content if state["idea"] else ""
        method_overview = state["method"].overview if state["method"] else ""
        results_summary = state["results"].summary if state["results"] else ""
        literature_synthesis = state["literature"].synthesis if state["literature"] else ""

        prompt = PAPER_WRITING_PROMPT.format(
            task=state["task"],
            literature=literature_synthesis,
            idea=idea_content,
            method=method_overview,
            results=results_summary
        )

        messages = [
            {"role": "system", "content": writer.system_prompt},
            {"role": "user", "content": prompt}
        ]

        log_stage(workspace_dir, "report_generation", "Generating paper")
        paper = llm_client.generate(messages)

        paper_path = get_artifact_path(workspace_dir, "paper")
        save_markdown(paper, paper_path.with_suffix('.md'))

        log_stage(workspace_dir, "report_generation", "Report generation completed")

        return {
            "paper": paper,
            "stage": "report_generation"
        }

    except Exception as e:
        log_stage(workspace_dir, "report_generation", f"Error: {str(e)}")
        raise WorkflowError(f"Report generation failed: {str(e)}")
