from researcher.state import ResearchState

# task router
def router_node(state: ResearchState) -> str:

    if state['task'] == None or state['task'] == "task_parsing":
        return "task_parsing"
    elif state['task'] == "literature_review":
        return "literature_review"
    elif state['task'] == "hypothesis_construction":
        return "hypothesis_construction"
    elif state['task'] == "method_design":
        return "method_design"
    elif state['task'] == "experiment_execution":
        return "experiment_execution"
    elif state['task'] == "report_generation":
        return "report_generation"
    elif state['task'] == "review":
        return "review"
    else:
        raise Exception(f"Unknown task: {state['task']}")