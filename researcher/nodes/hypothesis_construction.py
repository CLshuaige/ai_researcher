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
from researcher.schemas import ResearchIdea, IdeaCandidate
from researcher.agents import IdeaProposerAgent, IdeaCriticAgent, IdeaFormatterAgent
from researcher.utils import (
    save_markdown,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    parse_json_from_response,
    save_agent_history,
    iterable_group_chat
)
from researcher.prompts.templates import IDEA_PROPOSAL_PROMPT, IDEA_FORMATTER_PROMPT
from researcher.exceptions import WorkflowError


def hypothesis_construction_node(state: ResearchState) -> Dict[str, Any]:
    """Construct research hypothesis through multi-agent debate"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "hypothesis_construction", "Starting hypothesis construction")

    try:
        task = load_artifact_from_file(workspace_dir, "task")
        literature_text = load_artifact_from_file(workspace_dir, "literature") or "No literature review available"

        if not task:
            raise WorkflowError("Task file not found")

        config = state["config"]["researcher"]["hypothesis_construction"]
        max_iterations = config["max_debate_iterations"]

        llm_config = get_llm_config()

        proposer = IdeaProposerAgent().create_agent(llm_config)
        critic = IdeaCriticAgent().create_agent(llm_config)
        formatter = IdeaFormatterAgent().create_agent(llm_config)

        initial_context = ContextVariables(data={
                "debate_count": 0,
                "max_iterations": max_iterations
            }
        )

        pattern = DefaultPattern(
            initial_agent=proposer,
            agents=[proposer, critic, formatter],
            context_variables=initial_context,
            group_manager_args={"llm_config": llm_config}
        )

        proposer.handoffs.set_after_work(AgentTarget(critic))

        def format_debate_for_formatter(output, context_variables):
            history_parts = []
            for agent, name in [(proposer, "Proposer"), (critic, "Critic")]:
                chat_messages = getattr(agent, 'chat_messages', {})
                for message_list in chat_messages.values():
                    for msg in message_list:
                        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                            history_parts.append(f"[{name}]: {msg['content']}")
            
            debate_history = "\n\n".join(history_parts) if history_parts else "No debate history available."
            message = IDEA_FORMATTER_PROMPT.format(debate_history=debate_history)
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

            # Default: continue debate with proposer
            return FunctionTargetResult(
                target=AgentTarget(proposer),
                context_variables=ctx
            )

        critic.handoffs.set_after_work(FunctionTarget(critic_after_work))
        formatter.handoffs.set_after_work(TerminateTarget())

        initial_message = IDEA_PROPOSAL_PROMPT.format(task=task, literature=literature_text)
        log_stage(workspace_dir, "hypothesis_construction", f"Running debate (max {max_iterations} rounds)")

        if state["config"]["researcher"]["iterable"]:
            result, context, last_agent = iterable_group_chat(
                state,
                max_rounds=max_iterations * 2 + 1,
                enable_hitl=False,
                pattern=pattern,
                prompt=initial_message,
            )
        else:
            result, context, last_agent = initiate_group_chat(
                pattern=pattern,
                messages=initial_message,
                max_rounds=max_iterations * 2 + 1
            )

        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="hypothesis_construction",
            messages=result.chat_history,
            agent_chat_messages={
                proposer.name: proposer.chat_messages,
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

        debate_rounds = len([msg for msg in result.chat_history if msg.get("name") in [proposer.name, critic.name]]) // 2
        idea = _parse_idea(formatted_output, debate_rounds)

        idea_path = get_artifact_path(workspace_dir, "idea")
        save_markdown(idea.to_markdown(), idea_path)

        log_stage(workspace_dir, "hypothesis_construction", f"Completed. Generated {len(idea.candidates)} ideas")

        update_state = {
            "idea": idea,
            "stage": "hypothesis_construction"
        }
        # router
        if state["config"]["researcher"]["workflow"] == "default":
            update_state["next_node"] = "method_design"
        return update_state

    except Exception as e:
        log_stage(workspace_dir, "hypothesis_construction", f"Error: {str(e)}")
        raise WorkflowError(f"Hypothesis construction failed: {str(e)}")


def _parse_idea(formatted_output: Dict[str, Any], debate_rounds: int) -> ResearchIdea:
    candidates = []
    for idea_data in formatted_output.get("ideas", []):
        candidate = IdeaCandidate(
            content=idea_data.get("content", ""),
            score=idea_data.get("score", 0.0),
            round=idea_data.get("round", 0),
            strengths=idea_data.get("strengths", []),
            weaknesses=idea_data.get("weaknesses", [])
        )
        candidates.append(candidate)

    return ResearchIdea(
        candidates=candidates,
        selected_index=0,
        debate_rounds=debate_rounds
    )
