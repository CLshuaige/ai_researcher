from typing import Dict, Any

from researcher.state import ResearchState
from researcher.agents import WriterAgent
from researcher.config import get_model_config
from researcher.utils import save_markdown, log_stage, get_artifact_path, load_artifact_from_file
from researcher.prompts.templates import PAPER_WRITING_PROMPT
from researcher.llm import get_llm_client
from researcher.exceptions import WorkflowError


def report_generation_node(state: ResearchState) -> Dict[str, Any]:
    """Generate research report"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "report_generation", "Starting report generation")

    try:
        task = load_artifact_from_file(workspace_dir, "task") or ""
        literature_synthesis = load_artifact_from_file(workspace_dir, "literature") or ""
        idea_content = load_artifact_from_file(workspace_dir, "idea") or ""
        method_overview = load_artifact_from_file(workspace_dir, "method") or ""
        results_summary = load_artifact_from_file(workspace_dir, "results") or ""

        writer = WriterAgent()
        llm_client = get_llm_client(get_model_config())

        prompt = PAPER_WRITING_PROMPT.format(
            task=task,
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
            "task": task,
            "paper": paper,
            "stage": "report_generation"
        }

    except Exception as e:
        log_stage(workspace_dir, "report_generation", f"Error: {str(e)}")
        raise WorkflowError(f"Report generation failed: {str(e)}")
