from researcher.state import ResearchState
from langgraph.graph import END


def router_node(state: ResearchState) -> str:
    valid_nodes = {
        "task_parsing",
        "literature_review",
        "hypothesis_construction",
        "method_design",
        "experiment_execution",
        "report_generation",
        "review",
    }

    if state["stage"] == "initialization":
        start_node = state.get("start_node")
        if start_node is None or start_node == "task_parsing":
            return "task_parsing"
        if start_node in valid_nodes:
            return start_node
        raise Exception(f"Unknown node: {start_node}")
        
    next_node = state.get("next_node")
    if next_node in valid_nodes:
        return next_node
    if next_node in (None, "", "end"):
        return END
