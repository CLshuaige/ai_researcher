from typing import Dict, Any
from pathlib import Path

from autogen import GroupChat, GroupChatManager, UserProxyAgent

from researcher.state import ResearchState
from researcher.schemas import ExperimentalMethod, TaskAssignment
from researcher.agents import MethodPlannerAgent, MethodCriticAgent, MethodFormatterAgent
from researcher.config import DEBATE_MAX_ROUNDS
from researcher.utils import (
    save_markdown,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    parse_json_from_response,
    save_agent_history,
)
from researcher.prompts.templates import METHOD_PROPOSAL_PROMPT, METHOD_FORMATTER_PROMPT
from researcher.exceptions import WorkflowError


def method_design_node(state: ResearchState) -> Dict[str, Any]:
    """Design experimental method through multi-agent debate"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "method_design", "Starting method design")

    try:
        task = load_artifact_from_file(workspace_dir, "task")
        idea_content = load_artifact_from_file(workspace_dir, "idea") or "No idea available"

        if not task:
            raise WorkflowError("Task file not found")

        llm_config = get_llm_config()

        planner = MethodPlannerAgent().create_assistant(llm_config)
        critic = MethodCriticAgent().create_assistant(llm_config)

        initial_message = METHOD_PROPOSAL_PROMPT.format(idea=idea_content, task=task)

        log_stage(workspace_dir, "method_design", f"Running debate (max {DEBATE_MAX_ROUNDS} rounds)")

        groupchat = GroupChat(
            agents=[planner, critic],
            messages=[],
            max_round=DEBATE_MAX_ROUNDS,
            speaker_selection_method="round_robin"
        )

        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        planner.initiate_chat(manager, message=initial_message)

        debate_history = _extract_history(groupchat.messages)
        _save_log(workspace_dir, debate_history, "method_debate")

        # Save complete AG2 history
        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="method_design",
            messages=groupchat.messages,
            agent_chat_messages={
                planner.name: planner.chat_messages,
                critic.name: critic.chat_messages
            }
        )

        log_stage(workspace_dir, "method_design", "Formatting and structuring method")
        formatted_output = _format_output(debate_history, llm_config)

        method = _parse_method(formatted_output, len(debate_history) // 2)

        method_path = get_artifact_path(workspace_dir, "method")
        save_markdown(method.to_markdown(), method_path)

        log_stage(workspace_dir, "method_design", f"Completed. {len(method.steps)} steps defined")

        return {"task": task, "method": method, "stage": "method_design"}

    except Exception as e:
        log_stage(workspace_dir, "method_design", f"Error: {str(e)}")
        raise WorkflowError(f"Method design failed: {str(e)}")


def _extract_history(messages: list) -> list[Dict[str, str]]:
    history = []
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg and "name" in msg:
            history.append({"role": msg["name"], "content": msg["content"]})
    return history


def _save_log(workspace_dir: Path, history: list[Dict[str, str]], log_name: str):
    log_dir = workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{log_name}.log", 'w', encoding='utf-8') as f:
        for entry in history:
            f.write(f"[{entry['role']}]\n{entry['content']}\n\n")


def _format_output(debate_history: list[Dict[str, str]], llm_config: Dict[str, Any]) -> Dict[str, Any]:
    history_text = "\n\n".join([f"**{e['role']}**: {e['content']}" for e in debate_history])
    prompt = METHOD_FORMATTER_PROMPT.format(debate_history=history_text)

    formatter = MethodFormatterAgent().create_assistant(llm_config)
    user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=0)
    user_proxy.initiate_chat(formatter, message=prompt)

    response = user_proxy.last_message()["content"]

    return parse_json_from_response(response)


def _parse_method(formatted_output: Dict[str, Any], debate_rounds: int) -> ExperimentalMethod:
    assignments = []
    for assignment_data in formatted_output.get("assignments", []):
        assignment = TaskAssignment(
            role=assignment_data.get("role", ""),
            tasks=assignment_data.get("tasks", []),
            dependencies=assignment_data.get("dependencies", [])
        )
        assignments.append(assignment)

    return ExperimentalMethod(
        overview=formatted_output.get("overview", ""),
        steps=formatted_output.get("steps", []),
        assignments=assignments,
        resources=formatted_output.get("resources", {}),
        debate_rounds=debate_rounds
    )
