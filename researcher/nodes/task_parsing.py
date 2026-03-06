from typing import Dict, Any

from autogen import ConversableAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import (
    OnCondition,
    StringLLMCondition,
    OnContextCondition,
    ExpressionContextCondition,
    ContextExpression,
    ContextVariables,
    FunctionTarget,
    FunctionTargetResult,
    AgentTarget,
    TerminateTarget,
)

from researcher.state import ResearchState
from researcher.agents import AskerAgent, TaskFormatterAgent
from researcher.prompts.templates import TASK_CLARIFICATION_PROMPT
from researcher.utils import (
    save_markdown,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    save_agent_history,
)
from researcher.exceptions import WorkflowError


def task_parsing_node(state: ResearchState) -> Dict[str, Any]:
    """Parse and clarify research task with optional human-in-the-loop"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "task_parsing", "Starting task parsing")

    try:
        input_text = load_artifact_from_file(workspace_dir, "input")
        if not input_text:
            raise WorkflowError("Input file not found")

        config = state["config"]["researcher"]["task_parsing"]
        enable_hitl = config["human_in_the_loop"]
        max_iterations = config["max_iterations"]

        llm_config = get_llm_config()

        asker = AskerAgent().create_agent(llm_config)
        formatter = TaskFormatterAgent().create_agent(llm_config)
        
        if enable_hitl:
            user = ConversableAgent(
                name="user",
                human_input_mode="ALWAYS"
            )
            
            def asker_after_work(output: Any, ctx: ContextVariables) -> FunctionTargetResult:
                content = str(output)

                current_count = ctx.get("clarification_count", 0)
                max_iter = ctx.get("max_iterations", 0)
                
                current_count += 1
                ctx["clarification_count"] = current_count

                if "==CLEAR==" in content and "==UNCLEAR==" not in content:
                    return FunctionTargetResult(
                        target=AgentTarget(formatter),
                        context_variables=ctx
                    )
                
                if current_count >= max_iter:
                    return FunctionTargetResult(
                        target=AgentTarget(formatter),
                        context_variables=ctx
                    )

                return FunctionTargetResult(
                    target=AgentTarget(user),
                    context_variables=ctx
                )
            
            agents_list = [asker, formatter, user]
            user_agent = user
            initial_context = ContextVariables(data={
                    "clarification_count": 0,
                    "max_iterations": max_iterations
                }
            )
        else:
            agents_list = [asker, formatter]
            user_agent = None
            initial_context = ContextVariables()

        

        pattern = DefaultPattern(
            initial_agent=asker,
            agents=agents_list,
            user_agent=user_agent,
            context_variables=initial_context,
            group_manager_args={"llm_config": llm_config}
        )

        if enable_hitl:
            user.handoffs.set_after_work(AgentTarget(asker))
            asker.handoffs.set_after_work(FunctionTarget(asker_after_work))
        else:
            asker.handoffs.set_after_work(AgentTarget(formatter))

        formatter.handoffs.set_after_work(TerminateTarget())

        prompt = TASK_CLARIFICATION_PROMPT.format(input_text=input_text)
        
        max_rounds = max_iterations * 2 + 1 if enable_hitl else 4
        
        result, context, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=prompt,
            max_rounds=max_rounds
        )

        # Extract task from formatter's last message
        task = None
        for msg in reversed(result.chat_history):
            if msg.get("name") == formatter.name and msg.get("content"):
                task = msg["content"]
                break

        if not task:
            raise WorkflowError("Formatter did not generate task description")

        task_path = get_artifact_path(workspace_dir, "task")
        save_markdown(task, task_path)

        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="task_parsing",
            messages=result.chat_history,
            agent_chat_messages={
                asker.name: asker.chat_messages,
                formatter.name: formatter.chat_messages
            }
        )


        log_stage(workspace_dir, "task_parsing", "Completed")

        update_state = {"task": task, "stage": "task_parsing"}
        # router
        if state["config"]["researcher"]["workflow"] == "default":
            update_state["next_node"] = "literature_review"
        return update_state

    except Exception as e:
        log_stage(workspace_dir, "task_parsing", f"Error: {str(e)}")
        raise WorkflowError(f"Task parsing failed: {str(e)}")
