from typing import Dict, Any

from autogen import GroupChat, GroupChatManager, UserProxyAgent

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

        prompt = PAPER_WRITING_PROMPT.format(
            task=task,
            literature=literature_synthesis,
            idea=idea_content,
            method=method_overview,
            results=results_summary
        )

        writer = WriterAgent().create_assistant(llm_config)
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )

        log_stage(workspace_dir, "report_generation", "Generating paper")

        groupchat = GroupChat(
            agents=[user_proxy, writer],
            messages=[],
            max_round=1,
            speaker_selection_method="round_robin"
        )
        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        user_proxy.initiate_chat(manager, message=prompt)

        paper = user_proxy.last_message()["content"]

        # Save AG2 history
        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="report_generation",
            messages=groupchat.messages,
            agent_chat_messages=writer.chat_messages
        )

        paper_path = get_artifact_path(workspace_dir, "paper")
        save_markdown(paper, paper_path.with_suffix('.md'))

        log_stage(workspace_dir, "report_generation", "Completed")

        return {
            "task": task,
            "paper": paper,
            "stage": "report_generation"
        }

    except Exception as e:
        log_stage(workspace_dir, "report_generation", f"Error: {str(e)}")
        raise WorkflowError(f"Report generation failed: {str(e)}")
