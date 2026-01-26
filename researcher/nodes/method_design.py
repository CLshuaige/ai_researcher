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
    TerminateTarget,
    ContextVariables,
)

from researcher.state import ResearchState
from researcher.schemas import ExperimentalMethod, TaskAssignment, MethodStep
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
    """Design experimental method through multi-agent debate"""
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

        initial_context = ContextVariables(data={
                "debate_count": 0,
                "max_iterations": max_iterations
            }
        )

        pattern = DefaultPattern(
            initial_agent=planner,
            agents=[planner, critic, formatter],
            context_variables=initial_context,
            group_manager_args={"llm_config": llm_config}
        )

        planner.handoffs.set_after_work(AgentTarget(critic))

        def format_debate_for_formatter(output, context_variables):
            history_parts = []
            for agent, name in [(planner, "Planner"), (critic, "Critic")]:
                chat_messages = getattr(agent, 'chat_messages', {})
                for message_list in chat_messages.values():
                    for msg in message_list:
                        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                            history_parts.append(f"[{name}]: {msg['content']}")

            debate_history = "\n\n".join(history_parts) if history_parts else "No debate history available."
            debate_rounds = context_variables.get("debate_count", 0)
            message = METHOD_FORMATTER_PROMPT.format(debate_history=debate_history, debate_rounds=debate_rounds)
            return FunctionTargetResult(
                target=AgentTarget(formatter),
                messages=message,
                context_variables=context_variables
            )

        def critic_after_work(output: Any, ctx: ContextVariables) -> FunctionTargetResult:
            content = str(output)
            
            current_count = ctx.get("debate_count", 0)
            max_iter = ctx.get("max_iterations", max_iterations)
            
            current_count += 1
            ctx["debate_count"] = current_count

            # Check for READY identifier
            if "==READY==" in content and "==NEEDS_REVISION==" not in content:
                return FunctionTargetResult(
                    target=FunctionTarget(format_debate_for_formatter),
                    context_variables=ctx
                )
            
            # Check for iteration limit
            if current_count >= max_iter:
                return FunctionTargetResult(
                    target=FunctionTarget(format_debate_for_formatter),
                    context_variables=ctx
                )

            # Default: continue debate with planner
            return FunctionTargetResult(
                target=AgentTarget(planner),
                context_variables=ctx
            )

        critic.handoffs.set_after_work(FunctionTarget(critic_after_work))
        formatter.handoffs.set_after_work(TerminateTarget())

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
            messages=result.chat_history,
            agent_chat_messages={
                planner.name: planner.chat_messages,
                critic.name: critic.chat_messages,
                formatter.name: formatter.chat_messages
            }
        )

        # Extract formatted output from formatter
        formatted_output = None
        for msg in reversed(result.chat_history):
            if msg.get("name") == formatter.name and msg.get("content"):
                try:
                    formatted_output = parse_json_from_response(msg["content"])
                    break
                except:
                    pass

        if not formatted_output:
            raise WorkflowError("Formatter did not generate valid output")

        # Calculate debate rounds from message count (each round = 2 messages: planner + critic)
        debate_rounds = len([msg for msg in result.chat_history if msg.get("name") in [planner.name, critic.name]]) // 2
        method = _parse_method(formatted_output, debate_rounds)

        method_path = get_artifact_path(workspace_dir, "method")
        save_markdown(method.to_markdown(), method_path)

        log_stage(workspace_dir, "method_design", "Completed")

        return {"task": task, "method": method, "stage": "method_design"}

    except Exception as e:
        log_stage(workspace_dir, "method_design", f"Error: {str(e)}")
        raise WorkflowError(f"Method design failed: {str(e)}")


def _parse_method(formatted_output: Dict[str, Any], debate_rounds: int) -> ExperimentalMethod:
    steps = []
    for step_data in formatted_output.get("steps", []):
        step = MethodStep(
            step_id=step_data.get("step_id", 0),
            description=step_data.get("description", ""),
            assignee=step_data.get("assignee", ""),
            dependencies=step_data.get("dependencies", []),
            expected_output=step_data.get("expected_output", "")
        )
        steps.append(step)

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
        steps=steps,
        execution_order=formatted_output.get("execution_order", []),
        assignments=assignments,
        resources=formatted_output.get("resources", {}),
        debate_rounds=debate_rounds,
        criticisms=formatted_output.get("criticisms", [])
    )
