from typing import Dict, Any

from researcher.state import ResearchState
from researcher.utils import save_markdown, log_stage, get_artifact_path, load_artifact_from_file
from researcher.exceptions import WorkflowError


def task_parsing_node(state: ResearchState) -> Dict[str, Any]:
    """Parse and clarify research task"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "task_parsing", "Starting task parsing")

    try:
        input_text = load_artifact_from_file(workspace_dir, "input")
        if not input_text:
            raise WorkflowError("Input file not found")

        # TODO: Implement human-in-the-loop logic with Asker and Formatter agents
        # TODO: Determine if clarification is needed based on input quality

        task = input_text

        task_path = get_artifact_path(workspace_dir, "task")
        save_markdown(task, task_path)

        log_stage(workspace_dir, "task_parsing", "Task parsing completed")

        return {
            "task": task,
            "stage": "task_parsing"
        }

    except Exception as e:
        log_stage(workspace_dir, "task_parsing", f"Error: {str(e)}")
        raise WorkflowError(f"Task parsing failed: {str(e)}")
