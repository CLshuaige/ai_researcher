from datetime import datetime
import sys
from typing import Dict, Any

from autogen import ConversableAgent
from autogen.agentchat import initiate_group_chat, run_group_chat_iter
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
    RevertToUserTarget
)
from autogen.events.agent_events import GroupChatRunChatEvent, TextEvent, InputRequestEvent, TerminationEvent, RunCompletionEvent

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
<<<<<<< HEAD
import uuid
import asyncio


def _truncate_text(text: Any, limit: int = 500) -> str:
    raw = str(text)
    if len(raw) <= limit:
        return raw
    return f"{raw[:limit]}..."


def _publish_task_parsing_progress(
    state: ResearchState,
    progress_event: str,
    **extra: Any,
) -> None:
    """Publish per-iteration progress when API runtime is active."""
    project_id = state.get("project_id")
    if not project_id:
        return

    app_module = sys.modules.get("researcher.api.app")
    if app_module is None:
        return

    fastapi_app = getattr(app_module, "app", None)
    event_bus = None
    if fastapi_app is not None:
        event_bus = getattr(getattr(fastapi_app, "state", None), "event_bus", None)
    if event_bus is None:
        event_bus = getattr(app_module, "event_bus", None)
    if event_bus is None:
        return

    payload = {
        "event": "node_progress",
        "project_id": project_id,
        "node": "task_parsing",
        "stage": "task_parsing",
        "progress_event": progress_event,
        "timestamp": datetime.now().isoformat(),
    }
    payload.update(extra)

    try:
        event_bus.publish(project_id, payload)
    except Exception:
        # Progress publishing must not break workflow execution.
        pass
=======
>>>>>>> main


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
            #group_after_work=TerminateTarget(),
            group_manager_args={"llm_config": llm_config}
        )

        if enable_hitl:
            user.handoffs.set_after_work(AgentTarget(asker))
            asker.handoffs.set_after_work(FunctionTarget(asker_after_work))
        else:
            asker.handoffs.set_after_work(AgentTarget(formatter))

        formatter.handoffs.set_after_work(TerminateTarget())

        prompt = TASK_CLARIFICATION_PROMPT.format(input_text=input_text)
        
<<<<<<< HEAD
        max_rounds = max_iterations * 2 + 1 if enable_hitl else 2
        print("test")
        print(f"config: {state['config']}")
        print(f"iterable: {state['config']['researcher']['iterable']}")
=======
        max_rounds = max_iterations * 2 + 1 if enable_hitl else 4
>>>>>>> main
        
        if state["config"]["researcher"]["iterable"]:
            _publish_task_parsing_progress(
                state,
                "iterator_started",
                max_rounds=max_rounds,
                human_in_the_loop=enable_hitl,
            )
            iterator = run_group_chat_iter(
                pattern=pattern,
                messages=prompt,
                max_rounds=max_rounds,
                yield_on=[GroupChatRunChatEvent, TextEvent, InputRequestEvent, TerminationEvent, RunCompletionEvent]
            )

            global_history = []
            for step, event in enumerate(iterator, start=1):
                if isinstance(event, GroupChatRunChatEvent):
                    speaker = str(event.content.speaker)
                    print(f"\n=== {speaker}'s turn ===")
                    _publish_task_parsing_progress(
                        state,
                        "turn_started",
                        step=step,
                        speaker=speaker,
                    )
                elif isinstance(event, TextEvent):
                    sender = str(event.content.sender)
                    content = str(event.content.content)
                    print(content)
                    global_history.append({
                        "name": sender,
                        "content": content
                    })
                    _publish_task_parsing_progress(
                        state,
                        "message",
                        step=step,
                        sender=sender,
                        content_preview=_truncate_text(content),
                    )
                elif isinstance(event, InputRequestEvent):
                    prompt_text = str(event.content.prompt)
                    request_id = str(uuid.uuid4())
                    _publish_task_parsing_progress(
                        state,
                        "input_requested",
                        step=step,
                        request_id=request_id,
                        prompt=_truncate_text(prompt_text),
                    )
                    # api call
                    # 1. wait for client input
                    app_module = sys.modules.get("researcher.api.app")
                    print(f"app_module: {app_module}")

                    input_store = getattr(app_module, "input_store", None)
                    evt = input_store.create(request_id)
                    user_input = input_store.wait_for_input(request_id)  # 同步阻塞
                    # 2. input from cli
                    # user_input = input(prompt_text)
                    event.content.respond(user_input)
                    _publish_task_parsing_progress(
                        state,
                        "input_submitted",
                        step=step,
                        input_length=len(user_input),
                    )
                elif isinstance(event, TerminationEvent):
                    _publish_task_parsing_progress(
                        state,
                        "termination",
                        step=step,
                        detail=_truncate_text(event.content),
                    )
                elif isinstance(event, RunCompletionEvent):
                    result_history = event.content.history
                    summary = event.content.summary
                    context_vars = event.content.context_variables
                    last_speaker = event.content.last_speaker
                    _publish_task_parsing_progress(
                        state,
                        "run_completion",
                        step=step,
                        history_length=len(result_history),
                        last_speaker=str(last_speaker) if last_speaker else None,
                        summary_preview=_truncate_text(summary),
                    )
                print(event)
        else:
            result, context, last_agent = initiate_group_chat(
                pattern=pattern,
                messages=prompt,
                max_rounds=max_rounds
            )
            global_history = result.chat_history


        # print(f"global_history: {global_history}")
        # Extract task from formatter's last message
        task = None
        for msg in reversed(global_history):
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
            messages=global_history,
            agent_chat_messages={
                asker.name: asker.chat_messages,
                formatter.name: formatter.chat_messages
            }
        )


        log_stage(workspace_dir, "task_parsing", "Completed")
        _publish_task_parsing_progress(
            state,
            "completed",
            message_count=len(global_history),
        )

        update_state = {"task": task, "stage": "task_parsing"}
        # router
        if state["config"]["researcher"]["workflow"] == "default":
            update_state["next_node"] = "literature_review"
        return update_state

    except Exception as e:
        log_stage(workspace_dir, "task_parsing", f"Error: {str(e)}")
        _publish_task_parsing_progress(
            state,
            "failed",
            error=_truncate_text(e),
        )
        raise WorkflowError(f"Task parsing failed: {str(e)}")
