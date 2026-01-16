from typing import Dict, Any

from researcher.state import ResearchState
from researcher.schemas import ExperimentResult
#from researcher.config import config
from researcher.utils import save_markdown, log_stage, get_artifact_path
from researcher.exceptions import WorkflowError


def experiment_execution_node(state: ResearchState) -> Dict[str, Any]:
    """Execute experiments through multi-agent collaboration"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "experiment_execution", "Starting experiment execution")

    try:
        method = state["method"]

        results = ExperimentResult(
            summary="[TODO: Implement AG2-based experiment execution with RA, Engineer, and Analyst]",
            data_paths=[],
            figure_paths=[],
            metrics={},
            analysis=""
        )

        results_path = get_artifact_path(workspace_dir, "results")
        save_markdown(results.to_markdown(), results_path)

        log_stage(workspace_dir, "experiment_execution", "Experiment execution completed")

        return {
            "results": results,
            "stage": "experiment_execution"
        }

    except Exception as e:
        log_stage(workspace_dir, "experiment_execution", f"Error: {str(e)}")
        raise WorkflowError(f"Experiment execution failed: {str(e)}")
