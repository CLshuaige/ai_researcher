from typing import Dict, Any

from researcher.state import ResearchState
from researcher.schemas import ExperimentalMethod, TaskAssignment
from researcher.agents import PlannerAgent, CriticAgent
from researcher.debate import DebateTeam
from researcher.config import config
from researcher.utils import save_markdown, log_stage, get_artifact_path
from researcher.prompts.templates import METHOD_PROPOSAL_PROMPT
from researcher.exceptions import WorkflowError


def method_design_node(state: ResearchState) -> Dict[str, Any]:
    """Design experimental method through multi-agent debate"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "method_design", "Starting method design")

    try:
        idea_content = state["idea"].selected_idea.content if state["idea"] else "No idea available"

        initial_message = METHOD_PROPOSAL_PROMPT.format(
            idea=idea_content,
            task=state["task"]
        )

        proposer = PlannerAgent(name="MethodPlanner")
        critic = CriticAgent(name="MethodCritic")

        debate_team = DebateTeam(
            proposer=proposer,
            critic=critic,
            max_rounds=config.debate.max_rounds,
            workspace_dir=workspace_dir
        )

        log_stage(workspace_dir, "method_design", f"Running debate (max {config.debate.max_rounds} rounds)")
        debate_result = debate_team.run(initial_message)

        method = _parse_method_from_debate(debate_result.final_output, debate_result.rounds)

        method_path = get_artifact_path(workspace_dir, "method")
        save_markdown(method.to_markdown(), method_path)

        log_stage(workspace_dir, "method_design", f"Method design completed. {len(method.steps)} steps defined")

        return {
            "method": method,
            "stage": "method_design",
            "current_round": debate_result.rounds
        }

    except Exception as e:
        log_stage(workspace_dir, "method_design", f"Error: {str(e)}")
        raise WorkflowError(f"Method design failed: {str(e)}")


def _parse_method_from_debate(formatted_output: Dict[str, Any], debate_rounds: int) -> ExperimentalMethod:
    """Parse FormatterAgent output into ExperimentalMethod object"""

    assignments = []
    for assignment_data in formatted_output.get("assignments", []):
        assignment = TaskAssignment(
            role=assignment_data.get("role", ""),
            tasks=assignment_data.get("tasks", []),
            dependencies=assignment_data.get("dependencies", [])
        )
        assignments.append(assignment)

    method = ExperimentalMethod(
        overview=formatted_output.get("overview", ""),
        steps=formatted_output.get("steps", []),
        assignments=assignments,
        resources=formatted_output.get("resources", {}),
        debate_rounds=debate_rounds
    )

    return method
