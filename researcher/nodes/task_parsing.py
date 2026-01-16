from typing import Dict, Any

from autogen import GroupChat, GroupChatManager, UserProxyAgent

from researcher.state import ResearchState
from researcher.agents import AskerAgent, TaskFormatterAgent
from researcher.prompts.templates import TASK_CLARIFICATION_PROMPT
from researcher.utils import (
    save_markdown,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    save_agent_history,
    load_global_config,
)
from researcher.exceptions import WorkflowError


def task_parsing_node(state: ResearchState) -> Dict[str, Any]:
    """Parse and clarify research task with optional human-in-the-loop"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "task_parsing", "Starting task parsing")

    try:
        input_text = load_artifact_from_file(workspace_dir, "input")
        if not input_text:
            raise WorkflowError("Input file not found")

        global_config = load_global_config()
        task_parsing_config = global_config.get("researcher", {}).get("task_parsing", {})
        enable_hitl = task_parsing_config.get("human_in_the_loop", False)
        max_iterations = task_parsing_config.get("max_iterations", 3)

        llm_config = get_llm_config()

        asker = AskerAgent().create_assistant(llm_config)
        formatter = TaskFormatterAgent().create_assistant(llm_config)

        # Determine whether to use human in the loop
        if enable_hitl:
            user_proxy = UserProxyAgent(
                name="human",
                human_input_mode="ALWAYS",
                max_consecutive_auto_reply=0,
                code_execution_config=False
            )
            log_stage(workspace_dir, "task_parsing", "Human-in-the-loop enabled")
        else:
            user_proxy = UserProxyAgent(
                name="auto_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False
            )
            log_stage(workspace_dir, "task_parsing", "Auto mode (no human input)")

        # Phase 1: Asker analyzes input and asks clarifying questions
        groupchat = GroupChat(
            agents=[user_proxy, asker],
            messages=[],
            max_round=max_iterations,
            speaker_selection_method="round_robin"
        )

        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
        prompt = TASK_CLARIFICATION_PROMPT.format(input_text=input_text)
        user_proxy.initiate_chat(manager, message=prompt)

        conversation_history = _extract_conversation(groupchat.messages)

        # Phase 2: Formatter generates final task description
        # Use a separate user_proxy with NEVER mode for formatting (no human input needed)
        formatter_proxy = UserProxyAgent(
            name="formatter_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )

        formatter_prompt = f"""Based on the following conversation, format a clear and comprehensive research task description:

Original Input:
{input_text}

Conversation History:
{conversation_history}

Provide a well-structured task description that incorporates all clarifications."""

        groupchat_format = GroupChat(
            agents=[formatter_proxy, formatter],
            messages=[],
            max_round=1,
            speaker_selection_method="round_robin"
        )
        manager_format = GroupChatManager(groupchat=groupchat_format, llm_config=llm_config)
        formatter_proxy.initiate_chat(manager_format, message=formatter_prompt)
        task = formatter_proxy.last_message()["content"]

        task_path = get_artifact_path(workspace_dir, "task")
        save_markdown(task, task_path)

        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="task_parsing",
            messages=groupchat.messages + groupchat_format.messages,
            agent_chat_messages={
                asker.name: asker.chat_messages,
                formatter.name: formatter.chat_messages
            }
        )

        log_stage(workspace_dir, "task_parsing", "Task parsing completed")

        return {
            "task": task,
            "stage": "task_parsing"
        }

    except Exception as e:
        log_stage(workspace_dir, "task_parsing", f"Error: {str(e)}")
        raise WorkflowError(f"Task parsing failed: {str(e)}")


def _extract_conversation(messages: list) -> str:
    """Extract conversation history as formatted text"""
    conversation = []
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg and "name" in msg:
            role = msg["name"]
            content = msg["content"]
            conversation.append(f"[{role}]: {content}")
    return "\n\n".join(conversation)
