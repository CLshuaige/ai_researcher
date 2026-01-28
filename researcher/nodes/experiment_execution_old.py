from typing import Dict, Any, List
from pathlib import Path
import re

from autogen import ConversableAgent
from autogen.agentchat import initiate_group_chat
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.code_utils import create_virtual_env
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import (
    ContextVariables,
    AgentTarget,
    FunctionTarget,
    FunctionTargetResult,
    TerminateTarget,
)
from autogen.agentchat.contrib.capabilities import transform_messages, transforms

from researcher.state import ResearchState
from researcher.schemas import ExperimentResult, MethodStep, ResearchIdea
from researcher.agents import RAAgent, EngineerAgent, AnalystAgent
from researcher.utils import (
    save_markdown,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    save_agent_history,
)
from researcher.prompts.templates import RESULT_ANALYSIS_PROMPT, ENGINEER_STEP_PROMPT, RA_STEP_PROMPT
from researcher.exceptions import WorkflowError


def extract_selected_idea(idea_content: str) -> str:
    import re

    # Look for **[SELECTED]** marker
    selected_pattern = r'## Idea \d+ \*\*\[SELECTED\]\*\*\nScore: [\d.]+\nRound: \d+\n\n(.*?)(?=\n\n##|\n\n###|\Z)'
    match = re.search(selected_pattern, idea_content, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        # Fallback: return the first idea if no selection found
        first_idea_pattern = r'## Idea 1.*?\n\n(.*?)(?=\n\n##|\n\n###|\Z)'
        match = re.search(first_idea_pattern, idea_content, re.DOTALL)
        return match.group(1).strip() if match else idea_content


def parse_method_markdown(method_md: str) -> tuple[List[MethodStep], List[int]]:
    """Parse method markdown to extract structured steps and execution order"""
    steps = []
    execution_order = []

    # Extract execution order
    order_match = re.search(r'## Execution Order\n(.+)', method_md)
    if order_match:
        order_str = order_match.group(1).strip()
        execution_order = [int(x.strip()) for x in order_str.replace('→', ' ').split() if x.strip().isdigit()]

    # Extract steps
    step_pattern = r'### Step (\d+): (.+?)\n- \*\*Assignee\*\*: (.+?)\n(?:- \*\*Dependencies\*\*: (.+?)\n)?(?:- \*\*Expected Output\*\*: (.+?)\n)?'
    for match in re.finditer(step_pattern, method_md, re.MULTILINE):
        step_id = int(match.group(1))
        description = match.group(2).strip()
        assignee = match.group(3).strip()
        dependencies_str = match.group(4)
        expected_output = match.group(5) or ""

        dependencies = []
        if dependencies_str:
            dependencies = [int(x.strip()) for x in dependencies_str.split(',') if x.strip().isdigit()]

        steps.append(MethodStep(
            step_id=step_id,
            description=description,
            assignee=assignee,
            dependencies=dependencies,
            expected_output=expected_output.strip() if expected_output else ""
        ))

    # Fallback: if no execution order found, use step_id order
    if not execution_order and steps:
        execution_order = [step.step_id for step in sorted(steps, key=lambda s: s.step_id)]

    return steps, execution_order


def experiment_execution_node(state: ResearchState) -> Dict[str, Any]:
    """Execute experiments through multi-agent collaboration"""
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "experiment_execution", "Starting experiment execution")

    try:
        task_content = load_artifact_from_file(workspace_dir, "task")
        idea_content = load_artifact_from_file(workspace_dir, "idea")
        method_content = load_artifact_from_file(workspace_dir, "method")

        if not method_content:
            raise WorkflowError("Method file not found")

        config = state["config"]["researcher"]["experiment_execution"]
        timeout = config["code_execution_timeout"]
        max_retries = config["code_execution_retries"]
        use_virtual_env = config["use_virtual_env"]
        virtual_env_path = Path(config["virtual_env_path"])

        llm_config = get_llm_config()
        exp_dir = workspace_dir / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)

        code_dir = exp_dir / "code"
        results_dir = exp_dir / "results"
        figures_dir = exp_dir / "figures"

        code_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        steps, execution_order = parse_method_markdown(method_content)

        if not steps:
            raise WorkflowError("No structured steps found in method")

        log_stage(workspace_dir, "experiment_execution", f"Parsed {len(steps)} steps, execution order: {execution_order}")

        # Setup virtual environment for code execution
        virtual_env_context = None
        if use_virtual_env:
            import os
            venv_path = os.path.expanduser(virtual_env_path)

            if os.path.exists(venv_path):
                log_stage(workspace_dir, "experiment_execution", f"Using existing environment at {venv_path}")
                virtual_env_context = create_virtual_env(venv_path)
            else:
                raise WorkflowError(f"Virtual environment not found: {venv_path}")

        code_executor = LocalCommandLineCodeExecutor(
            timeout=timeout,
            work_dir=str(code_dir),
            virtual_env_context=virtual_env_context
        )

        ra = RAAgent().create_agent(llm_config)
        engineer = EngineerAgent().create_agent(llm_config)
        analyst = AnalystAgent().create_agent(llm_config)

        # Apply message history limiting to prevent context explosion
        # Keep only recent messages within message count limit
        message_limiter = transforms.MessageHistoryLimiter(max_messages=5)
        transform_messages.TransformMessages(transforms=[message_limiter]).add_to_agent(engineer)
        transform_messages.TransformMessages(transforms=[message_limiter]).add_to_agent(ra)

        code_executor_agent = ConversableAgent(
            name="CodeExecutor",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config={"executor": code_executor}
        )

        steps_dict = {step.step_id: step for step in steps}

        def step_dispatcher(output, context_variables: ContextVariables) -> FunctionTargetResult:
            """Dispatch next step based on execution order and dependencies"""
            current_idx = context_variables.get("current_step_index", 0)
            exec_order = context_variables.get("execution_order", [])
            step_results = context_variables.get("step_results", {})

            # Check if all steps completed
            if current_idx >= len(exec_order):
                # Format results for analyst
                results_summary = []
                for step_id in exec_order:
                    result = step_results.get(step_id, {})
                    status = result.get("status", "unknown")
                    output_text = result.get("output", "")
                    results_summary.append(f"Step {step_id} ({status}): {output_text[:200]}")

                message = RESULT_ANALYSIS_PROMPT.format(
                    method=method_content,
                    results="\n\n".join(results_summary)
                )

                return FunctionTargetResult(
                    target=AgentTarget(analyst),
                    messages=message,
                    context_variables=context_variables
                )

            # Get next step
            step_id = exec_order[current_idx]
            step = steps_dict[step_id]

            # Check dependencies
            unmet_deps = []
            for dep_id in step.dependencies:
                dep_result = step_results.get(dep_id, {})
                if dep_result.get("status") != "success":
                    unmet_deps.append(dep_id)

            if unmet_deps:
                log_stage(workspace_dir, "experiment_execution", f"Step {step_id} has unmet dependencies: {unmet_deps}")
                # Try to find and execute unmet dependencies first
                for dep_id in unmet_deps:
                    if dep_id in steps_dict and dep_id not in [r.get("step_id") for r in step_results.values()]:
                        # Execute dependency first
                        dep_step = steps_dict[dep_id]
                        context_variables["pending_step_id"] = step_id
                        return _dispatch_to_agent(dep_step, context_variables, task_content, idea_content, step_results)

                # If dependencies failed, skip this step
                log_stage(workspace_dir, "experiment_execution", f"Skipping step {step_id} due to failed dependencies")
                step_results[step_id] = {"status": "skipped", "output": f"Skipped due to failed dependencies: {unmet_deps}"}
                context_variables["current_step_index"] = current_idx + 1
                return step_dispatcher(output, context_variables)

            # Dispatch to appropriate agent
            return _dispatch_to_agent(step, context_variables, task_content, idea_content, step_results)

        def _dispatch_to_agent(step: MethodStep, context_variables: ContextVariables, task: str, idea: str, step_results: dict) -> FunctionTargetResult:
            """Dispatch step to RA or Engineer based on assignee"""
            # Collect file paths from previous steps
            available_files = []
            dep_summaries = []
            for dep_id in step.dependencies:
                if dep_id in step_results:
                    dep_result = step_results[dep_id]
                    # Get truncated summary
                    dep_summary = dep_result.get('output', '')[:200]
                    dep_summaries.append(f"Step {dep_id}: {dep_summary}")

                    dep_files = dep_result.get('files', [])
                    available_files.extend(dep_files)

            context_str = "\n".join(dep_summaries) if dep_summaries else "No previous results"
            files_str = "\n".join([f"- {f}" for f in available_files]) if available_files else "No files available from previous steps"

            selected_idea_content = extract_selected_idea(idea)

            # Store step info in context for completion detection
            context_variables["current_step_id"] = step.step_id
            context_variables["current_step_description"] = step.description
            context_variables["current_step_expected_output"] = step.expected_output

            if step.assignee == "RA":
                prompt = RA_STEP_PROMPT.format(
                    step_id=step.step_id,
                    task=task,
                    idea=selected_idea_content,
                    description=step.description,
                    expected_output=step.expected_output,
                    context=context_str,
                    available_files=files_str
                )
                return FunctionTargetResult(
                    target=AgentTarget(ra),
                    messages=prompt,
                    context_variables=context_variables
                )
            else:  # Engineer
                prompt = ENGINEER_STEP_PROMPT.format(
                    step_id=step.step_id,
                    task=task,
                    idea=selected_idea_content,
                    description=step.description,
                    expected_output=step.expected_output,
                    available_files=files_str,
                    context=context_str
                )
                return FunctionTargetResult(
                    target=AgentTarget(engineer),
                    messages=prompt,
                    context_variables=context_variables
                )

        def process_step_result(output, context_variables: ContextVariables) -> FunctionTargetResult:
            """Process step execution result and move to next step"""
            current_idx = context_variables.get("current_step_index", 0)
            exec_order = context_variables.get("execution_order", [])
            step_results = context_variables.get("step_results", {})
            step_retries = context_variables.get("step_retries", {})

            step_id = exec_order[current_idx]
            step = steps_dict[step_id]

            output_text = str(output)
            execution_status = "success"

            # Check for execution errors
            if hasattr(output, 'exit_code') and output.exit_code != 0:
                execution_status = "failed"
                log_stage(workspace_dir, "experiment_execution",
                         f"Step {step_id} execution failed with exit code {output.exit_code}: {output_text[:200]}{'...' if len(output_text) > 200 else ''}")
            elif any(keyword in output_text.lower() for keyword in ["error", "exception", "failed", "traceback"]):
                execution_status = "failed"

            if execution_status == "success":
                exit_info = f" (exit: {output.exit_code})" if hasattr(output, 'exit_code') else ""
                log_stage(workspace_dir, "experiment_execution", f"[✓] Step {step_id} code executed{exit_info}")

                context_variables["last_execution_output"] = output_text
                context_variables["last_execution_exit_code"] = getattr(output, 'exit_code', 0)

                # Check if this is from CodeExecutor - if so, return to Engineer for review
                if step.assignee == "Engineer":
                    return FunctionTargetResult(
                        target=AgentTarget(engineer),
                        messages=f"Code execution completed. Output:\n{output_text}\n\nReview the results. If the step goal is achieved, provide your summary and end with '==STEP_COMPLETE=='. Otherwise, continue working.",
                        context_variables=context_variables
                    )
                else:
                    # RA steps complete immediately after execution
                    step_results[step_id] = {
                        "step_id": step_id,
                        "status": "success",
                        "output": output_text,
                        "exit_code": getattr(output, 'exit_code', 0),
                        "execution_time": getattr(output, 'execution_time', None)
                    }
                    context_variables["current_step_index"] = current_idx + 1
                    context_variables["step_results"] = step_results
                    return step_dispatcher(output, context_variables)
            else:
                # Handle failure with retry
                retry_count = step_retries.get(step_id, 0)
                error_details = f"Step {step_id} failed"
                if hasattr(output, 'exit_code'):
                    error_details += f" (exit code: {output.exit_code})"
                error_details += f" - Output: {output_text[:200]}{'...' if len(output_text) > 200 else ''}"

                if retry_count < max_retries:
                    step_retries[step_id] = retry_count + 1
                    context_variables["step_retries"] = step_retries
                    log_stage(workspace_dir, "experiment_execution",
                             f"[↻] Step {step_id} failed - retry {retry_count + 1}/{max_retries}: {error_details}")

                    # Return to Engineer/RA with error info for retry
                    if step.assignee == "Engineer":
                        return FunctionTargetResult(
                            target=AgentTarget(engineer),
                            messages=f"Previous execution failed:\n{output_text}\n\nPlease debug and try again.",
                            context_variables=context_variables
                        )
                    else:
                        return step_dispatcher(output, context_variables)
                else:
                    # Max retries reached, skip step
                    log_stage(workspace_dir, "experiment_execution",
                             f"[✗] Step {step_id} failed permanently - max retries reached: {error_details}")
                    step_results[step_id] = {
                        "step_id": step_id,
                        "status": "failed",
                        "output": output_text,
                        "exit_code": getattr(output, 'exit_code', None),
                        "retries": retry_count
                    }
                    context_variables["current_step_index"] = current_idx + 1
                    context_variables["step_results"] = step_results

            # Continue to next step
            return step_dispatcher(output, context_variables)

        def check_engineer_completion(output, context_variables: ContextVariables) -> FunctionTargetResult:
            """Check if Engineer has marked the step as complete"""
            current_idx = context_variables.get("current_step_index", 0)
            exec_order = context_variables.get("execution_order", [])
            step_results = context_variables.get("step_results", {})
            step_id = exec_order[current_idx]

            output_text = str(output)

            # Check for STEP_COMPLETE marker
            if "==STEP_COMPLETE==" in output_text:
                # Extract completion summary
                import re
                match = re.search(r'==========STEP_COMPLETE==========\s*\n*(.+?)(?:\n==|$)', output_text, re.DOTALL)
                summary = match.group(1).strip() if match else "Step completed"

                # Extract file paths from summary
                file_paths = re.findall(r'(?:\.\.\/(?:results|figures)\/[^\s,]+\.(?:csv|json|png|jpg|jpeg|h5|pth|ckpt))', summary)

                log_stage(workspace_dir, "experiment_execution", f"[✓] Step {step_id} completed: {summary[:100]}")

                # Store step result
                step_results[step_id] = {
                    "step_id": step_id,
                    "status": "success",
                    "output": summary[:200],  # Truncate to 200 chars
                    "files": file_paths
                }
                context_variables["step_results"] = step_results
                context_variables["current_step_index"] = current_idx + 1

                # Move to next step
                return step_dispatcher(output, context_variables)
            else:
                # Engineer wants to continue working - check if there's code to execute
                # If Engineer wrote code, send to CodeExecutor
                # Otherwise, prompt Engineer to continue
                if "```python" in output_text or "```" in output_text:
                    # Has code block, send to executor
                    return FunctionTargetResult(
                        target=AgentTarget(code_executor_agent),
                        messages=output_text,
                        context_variables=context_variables
                    )
                else:
                    # No code, ask Engineer to continue
                    return FunctionTargetResult(
                        target=AgentTarget(engineer),
                        messages="Please continue working on the step or mark it complete with ==========STEP_COMPLETE==========",
                        context_variables=context_variables
                    )

        initial_context = ContextVariables(data={
            "current_step_index": 0,
            "execution_order": execution_order,
            "steps": [step.model_dump() for step in steps],
            "step_results": {},
            "step_retries": {},
            "max_retries": max_retries,
            "exp_dir": str(exp_dir),
            "code_dir": str(code_dir),
            "results_dir": str(results_dir),
            "figures_dir": str(figures_dir)
        })

        # Register handoffs for multi-round workflow
        # RA: simple flow - execute and complete
        ra.handoffs.set_after_work(FunctionTarget(process_step_result))

        # Engineer: multi-round flow
        # Engineer -> check_engineer_completion (checks for STEP_COMPLETE or code)
        #   -> if code: CodeExecutor -> process_step_result -> Engineer (review)
        #   -> if STEP_COMPLETE: move to next step
        engineer.handoffs.set_after_work(FunctionTarget(check_engineer_completion))

        # CodeExecutor: return results to Engineer for review
        code_executor_agent.handoffs.set_after_work(FunctionTarget(process_step_result))

        # Analyst: terminate when analysis is done
        analyst.handoffs.set_after_work(TerminateTarget())

        pattern = DefaultPattern(
            initial_agent=ra if steps_dict[execution_order[0]].assignee == "RA" else engineer,
            agents=[ra, engineer, code_executor_agent, analyst],
            user_agent=code_executor_agent,
            context_variables=initial_context,
            group_manager_args={"llm_config": llm_config}
        )

        # Start execution with first step
        first_step = steps_dict[execution_order[0]]
        first_dispatch = _dispatch_to_agent(first_step, initial_context, task_content, idea_content, {})
        first_prompt = first_dispatch.messages

        log_stage(workspace_dir, "experiment_execution", "Starting step-by-step execution")

        result, context, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=first_prompt,
            max_rounds=len(execution_order) * (max_retries + 2) + 5
        )

        # Extract analysis
        analysis = None
        for msg in reversed(result.chat_history):
            if msg.get("name") == analyst.name and msg.get("content"):
                content = msg["content"]
                if len(content) > 100:
                    analysis = content
                    break

        if not analysis:
            analysis = "Experiment execution completed. See conversation history for details."

        # Collect result files
        data_paths = list(results_dir.glob("*.csv")) + list(results_dir.glob("*.json"))
        figure_paths = list(figures_dir.glob("*.png")) + list(figures_dir.glob("*.jpg")) + list(figures_dir.glob("*.jpeg"))

        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="experiment_execution",
            messages=result.chat_history,
            agent_chat_messages={
                ra.name: ra.chat_messages,
                engineer.name: engineer.chat_messages,
                analyst.name: analyst.chat_messages
            }
        )

        exp_result = ExperimentResult(
            summary=analysis,
            data_paths=[str(p.relative_to(workspace_dir)) for p in data_paths],
            figure_paths=[str(p.relative_to(workspace_dir)) for p in figure_paths],
            metrics=context.get("metrics", {}),
            analysis=analysis
        )

        results_path = get_artifact_path(workspace_dir, "results")
        save_markdown(exp_result.to_markdown(), results_path)

        log_stage(workspace_dir, "experiment_execution", f"Completed. Generated {len(data_paths)} data files, {len(figure_paths)} figures")

        return {
            "results": exp_result,
            "stage": "experiment_execution"
        }

    except Exception as e:
        log_stage(workspace_dir, "experiment_execution", f"Error: {str(e)}")
        raise WorkflowError(f"Experiment execution failed: {str(e)}")
