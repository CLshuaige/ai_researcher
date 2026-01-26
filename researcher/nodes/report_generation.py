from typing import Dict, Any

from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import RevertToUserTarget

from researcher.state import ResearchState
from researcher.agents import WriterAgent
from researcher.utils import (
    save_markdown,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    save_agent_history,
)
from researcher.prompts.templates import PAPER_WRITING_PROMPT
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

        llm_config = get_llm_config()

        writer = WriterAgent().create_agent(llm_config)

        pattern = DefaultPattern(
            initial_agent=writer,
            agents=[writer],
            group_manager_args={"llm_config": llm_config}
        )

        writer.handoffs.set_after_work(RevertToUserTarget())

        prompt = PAPER_WRITING_PROMPT.format(
            task=task,
            literature=literature_synthesis,
            idea=idea_content,
            method=method_overview,
            results=results_summary
        )

        result, context, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=prompt,
            max_rounds=1
        )

        # Extract paper from writer
        paper = None
        for msg in reversed(result.messages):
            if msg.get("name") == writer.name and msg.get("content"):
                paper = msg["content"]
                break

        if not paper:
            raise WorkflowError("Writer did not generate paper")

        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="report_generation",
            messages=result.chat_history,
            agent_chat_messages=writer.chat_messages
        )

        paper_path = get_artifact_path(workspace_dir, "paper")
        save_markdown(paper, paper_path)

        log_stage(workspace_dir, "report_generation", "Completed")

        return {"task": task, "paper": paper, "stage": "report_generation"}

    except Exception as e:
        log_stage(workspace_dir, "report_generation", f"Error: {str(e)}")
        raise WorkflowError(f"Report generation failed: {str(e)}")
