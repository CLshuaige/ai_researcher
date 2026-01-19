from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from researcher.state import ResearchState
from researcher.nodes import (
    # task nodes
    task_parsing_node,
    hypothesis_construction_node,
    method_design_node,
    literature_review_node,
    experiment_execution_node,
    report_generation_node,
    review_node,

    # special nodes
    init_node,
    router_node,

)


def build_researcher_graph() -> StateGraph:
    """Build the main research workflow graph"""
    workflow = StateGraph(ResearchState)

    # Define nodes
    workflow.add_node("initialization"      , init_node)
    workflow.add_node("task_parsing"        , task_parsing_node)
    workflow.add_node("literature_review"   , literature_review_node)
    workflow.add_node("hypothesis_construction", hypothesis_construction_node)
    workflow.add_node("method_design"       , method_design_node)
    workflow.add_node("experiment_execution", experiment_execution_node)
    workflow.add_node("report_generation"   , report_generation_node)
    workflow.add_node("review"              , review_node)

    # Define edges
    workflow.add_edge(START                 , "initialization")
    workflow.add_conditional_edges("initialization", router_node)
    workflow.add_edge("task_parsing"        , "literature_review")
    workflow.add_edge("literature_review"   , "hypothesis_construction")
    workflow.add_edge("hypothesis_construction", "method_design")
    workflow.add_edge("method_design"       , "experiment_execution")
    workflow.add_edge("experiment_execution", "report_generation")
    workflow.add_edge("report_generation"   , "review")
    workflow.add_edge("review"              , END)

    # Compile with checkpointer
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
