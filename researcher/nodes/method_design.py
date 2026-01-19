from typing import Dict, Any
from pathlib import Path

from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import (
    OnCondition,
    StringLLMCondition,
    AgentTarget,
    FunctionTarget,
    FunctionTargetResult,
    RevertToUserTarget,
)

from researcher.state import ResearchState
from researcher.schemas import ExperimentalMethod, TaskAssignment
from researcher.agents import MethodPlannerAgent, MethodCriticAgent, MethodFormatterAgent
from researcher.utils import (
    save_markdown,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    parse_json_from_response,
    save_agent_history,
)
from researcher.prompts.templates import METHOD_PROPOSAL_PROMPT, METHOD_FORMATTER_PROMPT
from researcher.exceptions import WorkflowError


def method_design_node(state: ResearchState) -> Dict[str, Any]:
    """Design experimental method using DefaultPattern with Planner-Critic debate"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "method_design", "Starting method design")

    try:
        task = load_artifact_from_file(workspace_dir, "task")
        idea_content = load_artifact_from_file(workspace_dir, "idea") or "No idea available"

        if not task:
            raise WorkflowError("Task file not found")

        config = state["config"]["researcher"]["method_design"]
        max_iterations = config["max_debate_iterations"]

        llm_config = get_llm_config()

        planner = MethodPlannerAgent().create_agent(llm_config)
        critic = MethodCriticAgent().create_agent(llm_config)
        formatter = MethodFormatterAgent().create_agent(llm_config)

        pattern = DefaultPattern(
            initial_agent=planner,
            agents=[planner, critic, formatter],
            group_manager_args={"llm_config": llm_config}
        )

        planner.handoffs.set_after_work(AgentTarget(critic))

        def format_debate_for_formatter(context_variables):
            history_parts = []
            for agent, name in [(planner, "Planner"), (critic, "Critic")]:
                for msg in getattr(agent, 'chat_messages', []):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        history_parts.append(f"[{name}]: {msg['content']}")
            
            debate_history = "\n\n".join(history_parts) if history_parts else "No debate history available."
            message = METHOD_FORMATTER_PROMPT.format(debate_history=debate_history)
            return FunctionTargetResult(
                target=AgentTarget(formatter),
                message=message,
                context_variables=context_variables
            )

        critic.handoffs.add_llm_conditions([
            OnCondition(
                target=FunctionTarget(format_debate_for_formatter),
                condition=StringLLMCondition(
                    prompt="""Transfer to Formatter when:
- The experimental method is complete, feasible, and scientifically rigorous
- All steps are clearly defined with proper task assignments
- Resource requirements are realistic and well-documented
- All major concerns have been addressed"""
                ),
            ),
            OnCondition(
                target=AgentTarget(planner),
                condition=StringLLMCondition(
                    prompt="""Transfer back to Planner when:
- The method needs significant revision or improvement
- Steps are unclear, incomplete, or impractical
- Resource constraints are not properly addressed
- Further refinement is required"""
                ),
            ),
        ])

        formatter.handoffs.set_after_work(RevertToUserTarget())

        initial_message = METHOD_PROPOSAL_PROMPT.format(idea=idea_content, task=task)
        log_stage(workspace_dir, "method_design", f"Running debate (max {max_iterations} rounds)")

        result, context, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=initial_message,
            max_rounds=max_iterations * 2 + 1
        )

        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="method_design",
            messages=result.messages,
            agent_chat_messages={
                planner.name: planner.chat_messages,
                critic.name: critic.chat_messages,
                formatter.name: formatter.chat_messages
            }
        )

        # Extract formatted output from formatter
        formatted_output = None
        for msg in reversed(result.messages):
            if msg.get("name") == formatter.name and msg.get("content"):
                try:
                    formatted_output = parse_json_from_response(msg["content"])
                    break
                except:
                    pass

        if not formatted_output:
            raise WorkflowError("Formatter did not generate valid output")

        # Calculate debate rounds from message count (each round = 2 messages: planner + critic)
        debate_rounds = len([msg for msg in result.messages if msg.get("name") in [planner.name, critic.name]]) // 2
        method = _parse_method(formatted_output, debate_rounds)

        method_path = get_artifact_path(workspace_dir, "method")
        save_markdown(method.to_markdown(), method_path)

        log_stage(workspace_dir, "method_design", "Completed")

        return {"task": task, "method": method, "stage": "method_design"}

    except Exception as e:
        log_stage(workspace_dir, "method_design", f"Error: {str(e)}")
        raise WorkflowError(f"Method design failed: {str(e)}")


def _parse_method(formatted_output: Dict[str, Any], debate_rounds: int) -> ExperimentalMethod:
    assignments = []
    for assignment_data in formatted_output.get("assignments", []):
        assignment = TaskAssignment(
            role=assignment_data.get("role", ""),
            tasks=assignment_data.get("tasks", []),
            dependencies=assignment_data.get("dependencies", [])
        )
        assignments.append(assignment)

    return ExperimentalMethod(
        overview=formatted_output.get("overview", ""),
        steps=formatted_output.get("steps", []),
        assignments=assignments,
        resources=formatted_output.get("resources", {}),
        debate_rounds=debate_rounds,
        criticisms=formatted_output.get("criticisms", [])
    )
