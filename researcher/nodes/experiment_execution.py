from typing import Dict, Any, List
from pathlib import Path
import re

from autogen.agentchat import initiate_group_chat
from autogen.coding import LocalCommandLineCodeExecutor, CodeBlock
from autogen.code_utils import create_virtual_env
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import (
    ContextVariables,
    AgentTarget,
    FunctionTarget,
    FunctionTargetResult,
    TerminateTarget,
)
from autogen.agentchat.contrib.capabilities import transform_messages
from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter, MessageTokenLimiter

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
    deduplicate_long_repeats,
)
from researcher.prompts.templates import (
    RESULT_ANALYSIS_PROMPT,
    ENGINEER_STEP_PROMPT,
    RA_STEP_PROMPT,
    ENGINEER_DEBUG_PROMPT,
)
from researcher.exceptions import WorkflowError


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

        # Create Engineer-specific LLM config with lower temperature for more deterministic code generation
        engineer_llm_config = dict(llm_config)
        if "config_list" in engineer_llm_config:
            for config in engineer_llm_config["config_list"]:
                config["temperature"] = 0.3

        # Create experiment directory with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = workspace_dir / "experiments" / f"code_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

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

        ra = RAAgent().create_agent(llm_config)
        engineer = EngineerAgent().create_agent(engineer_llm_config, enable_context_compression=False)
        analyst = AnalystAgent().create_agent(llm_config)

        # Limit Engineer's message history to preserve system prompt and recent exchanges,
        # avoiding destructive summarization of code that could make it non-executable
        engineer_history_limiter = MessageHistoryLimiter(
            max_messages = 6,
            keep_first_message = True,
        )
        engineer_token_limiter = MessageTokenLimiter(
            max_tokens=25000,
            max_tokens_per_message=10000,
            min_tokens=15000,
            model="gpt-4-32k"
        )

        transform_messages.TransformMessages(
            transforms=[
                engineer_history_limiter,
                engineer_token_limiter
            ]
        ).add_to_agent(engineer)
        
        # Deduplicate long repeats in Engineer's messages such as path results
        engineer.register_hook("process_message_before_send", deduplicate_long_repeats)

        bootstrap_executor = LocalCommandLineCodeExecutor(
            timeout=timeout,
            work_dir=str(exp_dir),
            virtual_env_context=virtual_env_context,
        )

        steps_dict = {step.step_id: step for step in steps}
        
        def extract_complete_code_block(message: str) -> str:
            """Extract complete code block from first ``` to last ```"""
            text = message.strip()

            # Find first ```
            first_fence = text.find("```")
            if first_fence == -1:
                return message

            # Find last ```
            last_fence = text.rfind("```")
            if last_fence == -1 or last_fence <= first_fence:
                return message

            # Extract from first ``` to last ```
            return text[first_fence:last_fence + 3]

        def execute_code_directly(code_text: str, executor: LocalCommandLineCodeExecutor) -> str:
            """Execute code directly by creating CodeBlock, bypassing message parsing"""
            # Extract code content from the code block
            if code_text.startswith("```"):
                code_start = code_text.find("\n", code_text.find("```")) + 1
                code_end = code_text.rfind("```")
                code_content = code_text[code_start:code_end].strip()
            else:
                code_content = code_text

            code_block = CodeBlock(code=code_content, language="python")
            result = executor.execute_code_blocks([code_block])

            if result.exit_code == 0:
                output = f"exitcode: {result.exit_code} (execution succeeded)\nCode output: {result.output}"
            else:
                output = f"exitcode: {result.exit_code} (execution failed)\nCode output: {result.output}"

            return output

        def sanitize_dir_name(name: str) -> str:
            """Convert step description to safe directory name"""
            import re
            name = re.sub(r'[^\w\s-]', '', name)
            name = re.sub(r'[-\s]+', '_', name)
            return name[:50].strip('_').lower() or 'step'
        
        def get_step_dir(step: MethodStep) -> Path:
            """Get unified experiment directory for all steps"""
            step_dir = exp_dir
            step_dir.mkdir(parents=True, exist_ok=True)
            try:
                import json
                meta_path = step_dir / f"step_{step.step_id}_meta.json"
                if not meta_path.exists():
                    meta = {
                        "step_id": step.step_id,
                        "description": step.description,
                        "assignee": step.assignee,
                        "dependencies": step.dependencies,
                        "expected_output": step.expected_output,
                        "work_dir": str(step_dir),
                    }
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                # Best-effort only; do not fail workflow due to metadata writing.
                pass
            return step_dir
        
        def _get_file_snapshot(step_dir: Path) -> set:
            """Get snapshot of all files in step directory before execution"""
            if not step_dir.exists():
                return set()
            all_files = set(step_dir.rglob("*"))
            return {f for f in all_files if f.is_file()}

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
                    results_summary.append(f"Step {step_id} ({status}): {output_text}")

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
                log_stage(workspace_dir, "experiment_execution", f"[!] Step {step_id} has unmet dependencies: {unmet_deps} (available: {list(step_results.keys())})")
                # Try to find and execute unmet dependencies first
                for dep_id in unmet_deps:
                    if dep_id in steps_dict and dep_id not in [r.get("step_id") for r in step_results.values()]:
                        # Execute dependency first
                        dep_step = steps_dict[dep_id]
                        context_variables["pending_step_id"] = step_id
                        return _dispatch_to_agent(dep_step, context_variables, task_content, idea_content, step_results)

                # If dependencies failed, skip this step
                log_stage(workspace_dir, "experiment_execution", f"[✗] Step {step_id} skipped due to failed dependencies")
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

            # Deduplicate and limit available files to prevent context length explosion
            # if available_files:
            #     unique_files = list(set(available_files))  # Remove duplicates
            #     max_files_display = 15  # Limit to prevent context overflow

            #     if len(unique_files) <= max_files_display:
            #         files_str = "\n".join([f"- {f}" for f in unique_files])
            #     else:
            #         files_str = f"Total {len(unique_files)} files available from previous steps.\n"
            #         files_str += "Showing first 15 files as examples:\n"
            #         files_str += "\n".join([f"- {f}" for f in unique_files[:15]])
            # else:
            #     files_str = "No files available from previous steps"

            selected_idea_content = extract_selected_idea(idea)

            # Store step info in context for completion detection
            context_variables["current_step_id"] = step.step_id
            context_variables["current_step_description"] = step.description
            context_variables["current_step_expected_output"] = step.expected_output
            
            step_dir = get_step_dir(step)
            context_variables[f"step_dir_{step.step_id}"] = str(step_dir)
            
            files_before = _get_file_snapshot(step_dir)
            context_variables[f"files_before_step_{step.step_id}"] = files_before
            
            step_code_executor = LocalCommandLineCodeExecutor(
                timeout=timeout,
                work_dir=str(step_dir),
                virtual_env_context=virtual_env_context
            )
            context_variables["step_code_executor"] = step_code_executor

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
                timestamp = exp_dir.name.replace("code_", "")
                prompt = ENGINEER_STEP_PROMPT.format(
                    step_id=step.step_id,
                    task=task,
                    idea=selected_idea_content,
                    description=step.description,
                    expected_output=step.expected_output,
                    available_files=files_str,
                    context=context_str,
                    timestamp=timestamp
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
            
            # Parse exit code from autogen's formatted string
            # Format: "exitcode: {exit_code} ({status})\nCode output: {output}"
            exit_code = None
            execution_status = "unknown"
            
            exitcode_match = re.search(r'exitcode:\s*(\d+)', output_text)
            
            if "execution succeeded" in output_text.lower():
                if exitcode_match:
                    exit_code = int(exitcode_match.group(1))
                    if exit_code == 0:
                        execution_status = "success"
                    else:
                        execution_status = "failed"
                        log_stage(workspace_dir, "experiment_execution",
                                 f"[✗] Step {step_id} execution failed with exit code {exit_code} despite 'succeeded' message: {output_text}")
                else:
                    execution_status = "success"
            elif "execution failed" in output_text.lower():
                execution_status = "failed"
                if exitcode_match:
                    exit_code = int(exitcode_match.group(1))
                log_stage(workspace_dir, "experiment_execution",
                         f"[✗] Step {step_id} execution failed: {output_text}")
            elif exitcode_match:
                exit_code = int(exitcode_match.group(1))
                execution_status = "success" if exit_code == 0 else "failed"
                if execution_status == "failed":
                    log_stage(workspace_dir, "experiment_execution",
                             f"[✗] Step {step_id} execution failed with exit code {exit_code}: {output_text}")
            elif any(keyword in output_text.lower() for keyword in ["error", "exception", "traceback"]):
                execution_status = "failed"
                log_stage(workspace_dir, "experiment_execution",
                         f"[✗] Step {step_id} execution likely failed (error keywords detected): {output_text}")

            if execution_status == "success":
                exit_info = f" (exit: {exit_code})" if exit_code is not None else ""
                log_stage(workspace_dir, "experiment_execution", f"[✓] Step {step_id} code executed{exit_info}")

                context_variables["last_execution_output"] = output_text
                context_variables["last_execution_exit_code"] = exit_code if exit_code is not None else 0
                
                step_dir = Path(context_variables.get(f"step_dir_{step_id}", exp_dir))
                log_file = step_dir / f"step_{step_id}_execution.log"
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(output_text)
                log_stage(workspace_dir, "experiment_execution", 
                         f"[✓] Step {step_id} execution log saved to {log_file.relative_to(workspace_dir)}")

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
                        "exit_code": exit_code if exit_code is not None else 0,
                        "execution_time": None  # LocalCommandLineCodeExecutor doesn't provide execution_time
                    }
                    context_variables["current_step_index"] = current_idx + 1
                    context_variables["step_results"] = step_results
                    return step_dispatcher(output, context_variables)
            elif execution_status == "unknown":
                log_stage(workspace_dir, "experiment_execution",
                         f"[?] Step {step_id} received message (not code execution result): {output_text}")
                if step.assignee == "RA":
                    step_results[step_id] = {
                        "step_id": step_id,
                        "status": "success",
                        "output": output_text,
                        "exit_code": 0,
                        "execution_time": None
                    }
                    context_variables["current_step_index"] = current_idx + 1
                    context_variables["step_results"] = step_results
                    return step_dispatcher(output, context_variables)
                return FunctionTargetResult(
                    target=AgentTarget(engineer),
                    messages=output_text,
                    context_variables=context_variables
                )
            else:
                # Handle failure with retry
                retry_count = step_retries.get(step_id, 0)
                error_details = f"Step {step_id} failed"
                if hasattr(output, 'exit_code'):
                    error_details += f" (exit code: {output.exit_code})"
                error_details += f" - Output: {output_text}"

                if retry_count < max_retries:
                    step_retries[step_id] = retry_count + 1
                    context_variables["step_retries"] = step_retries
                    log_stage(workspace_dir, "experiment_execution",
                             f"[↻] Step {step_id} failed - retry {retry_count + 1}/{max_retries}: {error_details}")

                    # Return to Engineer/RA with error info for retry
                    if step.assignee == "Engineer":
                        return FunctionTargetResult(
                            target=AgentTarget(engineer),
                            messages=ENGINEER_DEBUG_PROMPT.format(output_text=output_text),
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
            """Handle explicit step completion marked by the Engineer."""
            current_idx = context_variables.get("current_step_index", 0)
            exec_order = context_variables.get("execution_order", [])
            step_results = context_variables.get("step_results", {})
            step_id = exec_order[current_idx]

            output_text = str(output)

            # Check for STEP_COMPLETE marker only (code is handled elsewhere).
            if "==STEP_COMPLETE==" not in output_text:
                return FunctionTargetResult(
                    target=AgentTarget(engineer),
                    messages=(
                        "No STEP_COMPLETE marker found. Check the result files."
                        "If you confirm the result already exists and is correct, respond with either:\n\n"
                        "1) A Python code block to continue working, or\n"
                        "2) The completion format:\n"
                        "==========STEP_COMPLETE==========\n[brief summary]"
                    ),
                    context_variables=context_variables,
                )

            log_stage(workspace_dir, "experiment_execution", f"[✓] Step {step_id} marked as complete by Engineer")

            # Extract completion summary
            import re

            match = re.search(r'==========STEP_COMPLETE==========\s*\n*(.+?)(?:\n==|$)', output_text, re.DOTALL)
            summary = match.group(1).strip() if match else "Step completed"

            step_dir = Path(context_variables.get(f"step_dir_{step_id}", exp_dir))
            files_before = context_variables.get(f"files_before_step_{step_id}", set())
            files_after = _get_file_snapshot(step_dir)

            new_files = files_after - files_before
            detected_files = [str(f.relative_to(workspace_dir)) for f in new_files if f.is_file()]

            try:
                summary_path = step_dir / f"step_{step_id}_summary.md"
                summary_path.write_text(summary, encoding="utf-8")
            except Exception:
                # Completion should not fail because of summary persistence issues.
                pass

            if detected_files:
                log_stage(workspace_dir, "experiment_execution", f"[✓] Step {step_id} completed: {summary}, generated {len(detected_files)} files: {detected_files}")
            else:
                log_stage(workspace_dir, "experiment_execution", f"[!] Step {step_id} completed: {summary}, but no result files were detected")

            step_results[step_id] = {
                "step_id": step_id,
                "status": "success",
                "output": summary,
                "files": detected_files,
            }
            context_variables["step_results"] = step_results
            context_variables["current_step_index"] = current_idx + 1

            # Move to next step in the execution graph.
            return step_dispatcher(output, context_variables)

        def handle_engineer_response(output, context_variables: ContextVariables) -> FunctionTargetResult:
            """Route Engineer responses: prefer executing code blocks, otherwise check completion."""
            output_text = str(output).strip()

            # If the Engineer provided a code block, execute it first.
            if "```python" in output_text or "```" in output_text:
                complete_code_block = extract_complete_code_block(output_text)

                step_executor = context_variables.get("step_code_executor")
                if step_executor is None:
                    step_executor = bootstrap_executor  # Fallback

                execution_output = execute_code_directly(complete_code_block, step_executor)
                return process_step_result(execution_output, context_variables)

            # No code block – treat this as a completion attempt.
            return check_engineer_completion(output, context_variables)

        initial_context = ContextVariables(data={
            "current_step_index": 0,
            "execution_order": execution_order,
            "steps": [step.model_dump() for step in steps],
            "step_results": {},
            "step_retries": {},
            "max_retries": max_retries,
            "exp_dir": str(exp_dir)
        })

        # Register handoffs for multi-round workflow
        # RA: simple flow - execute and complete
        ra.handoffs.set_after_work(FunctionTarget(process_step_result))
        engineer.handoffs.set_after_work(FunctionTarget(handle_engineer_response))
        analyst.handoffs.set_after_work(TerminateTarget())

        pattern = DefaultPattern(
            initial_agent=ra if steps_dict[execution_order[0]].assignee == "RA" else engineer,
            agents=[ra, engineer, analyst],
            user_agent=engineer,
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

        # Extract analysis from analyst messages (no minimum length requirement).
        analyst_messages = [
            msg.get("content")
            for msg in result.chat_history
            if msg.get("name") == analyst.name and msg.get("content")
        ]

        analysis = analyst_messages[-1] if analyst_messages else None

        if not analysis:
            error_msg = "[✕] No analysis message found from analyst agent."
            log_stage(workspace_dir, "experiment_execution", error_msg)
            raise WorkflowError("[✕]Analyst analysis is required but not found. Check agent execution flow.")

        all_result_files = []
        for step_id in execution_order:
            step_result = context.get("step_results", {}).get(step_id, {})
            step_files = step_result.get("files", [])
            all_result_files.extend([Path(workspace_dir / f) for f in step_files if Path(workspace_dir / f).exists()])
        
        all_result_files = list(set(all_result_files))
        
        figure_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.gif', '.bmp', '.tiff'}
        data_files = [p for p in all_result_files if p.suffix.lower() not in figure_extensions]
        figure_files = [p for p in all_result_files if p.suffix.lower() in figure_extensions]
        
        log_stage(workspace_dir, "experiment_execution",
                 f"Final file collection: {len(data_files)} data files, {len(figure_files)} figure files")

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
            data_paths=[str(p.relative_to(workspace_dir)) for p in data_files],
            figure_paths=[str(p.relative_to(workspace_dir)) for p in figure_files],
            metrics=context.get("metrics", {})
        )

        results_path = get_artifact_path(workspace_dir, "results")
        save_markdown(exp_result.to_markdown(), results_path)

        log_stage(workspace_dir, "experiment_execution", f"Completed. Generated {len(data_files)} data files, {len(figure_files)} figure files")

        update_state = {
            "results": exp_result,
            "stage": "experiment_execution"
        }
        # router
        if state["config"]["researcher"]["workflow"] == "default":
            update_state["next_node"] = "report_generation"
        return update_state

    except Exception as e:
        log_stage(workspace_dir, "experiment_execution", f"Error: {str(e)}")
        raise WorkflowError(f"Experiment execution failed: {str(e)}")


def extract_selected_idea(idea_content: str) -> str:
    """Extract the selected research idea from idea content"""
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
