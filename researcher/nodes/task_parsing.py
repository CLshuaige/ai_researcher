from typing import Dict, Any

from autogen import UserProxyAgent

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

        llm_config = get_llm_config()

        # Check if human feedback is already provided (continuing from interrupt)
        human_feedback = state.get("human_feedback")

        if human_feedback:
            # Human has provided answers, format the final task
            log_stage(workspace_dir, "task_parsing", "Processing human feedback")
            task = _format_task_with_feedback(input_text, human_feedback, llm_config, workspace_dir)
        else:
            # First pass: check if clarification is needed
            asker = AskerAgent().create_assistant(llm_config)
            user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=0)

            prompt = TASK_CLARIFICATION_PROMPT.format(input_text=input_text)
            user_proxy.initiate_chat(asker, message=prompt)

            response = user_proxy.last_message()["content"]

            save_agent_history(
                workspace_dir=workspace_dir,
                node_name="task_parsing",
                messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                agent_chat_messages=asker.chat_messages
            )

            if response.startswith("CLEAR:"):
                # Task is clear, extract and save
                task = response.replace("CLEAR:", "").strip()
                log_stage(workspace_dir, "task_parsing", "Task is clear, no clarification needed")
            elif response.startswith("UNCLEAR:"):
                # Task needs clarification, prepare questions for human
                questions = response.replace("UNCLEAR:", "").strip()
                log_stage(workspace_dir, "task_parsing", "Task needs clarification, waiting for human input")

                # Store questions in human_feedback for the interrupt
                return {
                    "task": None,
                    "stage": "task_parsing",
                    "human_feedback": {
                        "type": "clarification",
                        "questions": questions,
                        "original_input": input_text
                    }
                }
            else:
                # Fallback: use input as-is
                task = input_text
                log_stage(workspace_dir, "task_parsing", "Unable to parse response, using input as-is")

        task_path = get_artifact_path(workspace_dir, "task")
        save_markdown(task, task_path)

        log_stage(workspace_dir, "task_parsing", "Task parsing completed")

        return {
            "task": task,
            "stage": "task_parsing",
            "human_feedback": None  # Clear feedback after processing
        }

    except Exception as e:
        log_stage(workspace_dir, "task_parsing", f"Error: {str(e)}")
        raise WorkflowError(f"Task parsing failed: {str(e)}")


def _format_task_with_feedback(
    input_text: str,
    human_feedback: Dict[str, Any],
    llm_config: Dict[str, Any],
    workspace_dir
) -> str:
    """Format final task using human feedback"""
    questions = human_feedback.get("questions", "")
    answers = human_feedback.get("answers", "")

    formatter = TaskFormatterAgent().create_assistant(llm_config)
    user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=0)

    prompt = f"""Format the following research task based on the original input and clarifications:

Original Input:
{input_text}

Questions Asked:
{questions}

User Answers:
{answers}

Provide a clear, comprehensive task description that incorporates all information."""

    user_proxy.initiate_chat(formatter, message=prompt)
    task = user_proxy.last_message()["content"]

    save_agent_history(
        workspace_dir=workspace_dir,
        node_name="task_parsing",
        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": task}],
        agent_chat_messages=formatter.chat_messages
    )

    return task
