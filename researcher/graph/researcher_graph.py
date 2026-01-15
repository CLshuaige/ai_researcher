from langgraph.graph import StateGraph, END

from researcher.state import ResearchState
from researcher.nodes import (
    task_parsing_node,
    hypothesis_construction_node,
    method_design_node,
    literature_review_node,
    experiment_execution_node,
    report_generation_node,
    review_node
)


def build_researcher_graph() -> StateGraph:
    """Build the main research workflow graph"""
    workflow = StateGraph(ResearchState)

    workflow.add_node("task_parsing"        , task_parsing_node)
    workflow.add_node("literature_review"   , literature_review_node)
    workflow.add_node("hypothesis_construction", hypothesis_construction_node)
    workflow.add_node("method_design"       , method_design_node)
    workflow.add_node("experiment_execution", experiment_execution_node)
    workflow.add_node("report_generation"   , report_generation_node)
    workflow.add_node("review"              , review_node)

    workflow.set_entry_point("task_parsing")
    workflow.add_edge("task_parsing"        , "literature_review")
    workflow.add_edge("literature_review"   , "hypothesis_construction")
    workflow.add_edge("hypothesis_construction", "method_design")
    workflow.add_edge("method_design"       , "experiment_execution")
    workflow.add_edge("experiment_execution", "report_generation")
    workflow.add_edge("report_generation"   , "review")
    workflow.add_edge("review"              , END)

    return workflow.compile()
