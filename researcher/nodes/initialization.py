from copy import deepcopy
from typing import Dict, Any

from researcher.state import ResearchState
from researcher.utils import log_stage, merge_dict, set_llm_config_override


def _normalize_run_mode(run_mode: str | None, workflow: str | None) -> tuple[str, str]:

    if run_mode:
        mode = run_mode.strip().lower()
    else:
        mode = "auto" if (workflow or "").strip().lower() == "default" else "step"

    if mode in {"auto", "default"}:
        return "auto", "default"
    return "step", "step"


def init_node(state: ResearchState) -> Dict[str, Any]:

    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "initialization", "Applying global runtime settings")

    config = deepcopy(state.get("config") or {})
    post_config = state.get("post_config") or {}
    if post_config:
        config = merge_dict(config, post_config)

    researcher_cfg = config.setdefault("researcher", {})
    workflow = researcher_cfg.get("workflow")
    run_mode, workflow_mode = _normalize_run_mode(state.get("run_mode"), workflow)
    researcher_cfg["workflow"] = workflow_mode

    llm_config = config.get("llm_config")
    set_llm_config_override(llm_config if isinstance(llm_config, dict) else None)

    project_id = state.get("project_id") or state.get("session_id")

    return {
        "stage": "initialization",
        "config": config,
        "project_id": project_id,
        "run_mode": run_mode,
    }
