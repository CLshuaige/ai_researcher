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
    RevertToUserTarget,
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
            
            def increment_clarification_count(context_variables: ContextVariables) -> FunctionTargetResult:
                current_count = context_variables.get("clarification_count", 0)
                context_variables["clarification_count"] = current_count + 1
                return FunctionTargetResult(
                    target=AgentTarget(asker),
                    context_variables=context_variables
                )
            
            agents_list = [asker, formatter, user]
            user_agent = user
            initial_context = ContextVariables(
                clarification_count=0,
                max_iterations=max_iterations
            )
        else:
            agents_list = [asker, formatter]
            user_agent = None
            initial_context = None

        pattern = DefaultPattern(
            initial_agent=asker,
            agents=agents_list,
            user_agent=user_agent,
            context_variables=initial_context,
            group_manager_args={"llm_config": llm_config}
        )

        if enable_hitl:
            asker.handoffs.add_llm_conditions([
                OnCondition(
                    target=AgentTarget(user),
                    condition=StringLLMCondition(
                        prompt="""Transfer to User when you need to ask clarifying questions:
- The research task is ambiguous or incomplete
- Critical information is missing (objectives, scope, constraints, resources)
- You need user input to better understand the requirements
- Further clarification is needed before proceeding"""
                    ),
                ),
                OnCondition(
                    target=AgentTarget(formatter),
                    condition=StringLLMCondition(
                        prompt="""Transfer to Formatter when the task is clear and no more clarification is needed:
- The research task is clearly defined with specific objectives
- All necessary information (scope, constraints, resources) is available
- You have gathered sufficient information from the user (or no clarification was needed)
- The task is ready to be formatted and finalized"""
                    ),
                ),
            ])
            asker.handoffs.add_context_conditions([
                OnContextCondition(
                    target=AgentTarget(formatter),
                    condition=ExpressionContextCondition(
                        expression=ContextExpression("${clarification_count} >= ${max_iterations}")
                    )
                ),
            ])
            user.handoffs.set_after_work(FunctionTarget(increment_clarification_count))
        else:
            asker.handoffs.set_after_work(AgentTarget(formatter))

        formatter.handoffs.set_after_work(RevertToUserTarget())

        prompt = TASK_CLARIFICATION_PROMPT.format(input_text=input_text)
        
        max_rounds = max_iterations * 2 + 1 if enable_hitl else 2
        
        result, context, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=prompt,
            max_rounds=max_rounds
        )

        # Extract task from formatter's last message
        task = None
        for msg in reversed(result.messages):
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
            messages=result.messages,
            agent_chat_messages={
                asker.name: asker.chat_messages,
                formatter.name: formatter.chat_messages
            }
        )

        log_stage(workspace_dir, "task_parsing", "Completed")
        return {"task": task, "stage": "task_parsing"}

    except Exception as e:
        log_stage(workspace_dir, "task_parsing", f"Error: {str(e)}")
        raise WorkflowError(f"Task parsing failed: {str(e)}")
