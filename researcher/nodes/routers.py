from researcher.state import ResearchState
from langgraph.graph import END

# task router
def router_node(state: ResearchState) -> str:

    if state["stage"] == "initialization":
        start_node = state["start_node"]
        if start_node == None or start_node == "task_parsing":
            return "task_parsing"
        elif start_node in ["literature_review", "hypothesis_construction", "method_design", "experiment_execution", "report_generation", "review"]:
            return start_node
        else:
            raise Exception(f"Unknown node: {start_node}")
        
    else:
        next_node = state["next_node"]
        if next_node in ["literature_review", "hypothesis_construction", "method_design", "experiment_execution", "report_generation", "review"]:
            return next_node
        elif next_node == "end":
            return END