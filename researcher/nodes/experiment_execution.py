from typing import Dict, Any, List, Optional
import atexit
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from autogen import ConversableAgent
from autogen.agentchat import initiate_group_chat
from autogen.coding import LocalCommandLineCodeExecutor, CodeBlock
from autogen.code_utils import create_virtual_env
from autogen.agentchat.group.patterns import DefaultPattern
from autogen_agentchat.messages import TextMessage
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
from researcher.agents import RAAgent, EngineerAgent, CodeDebuggerAgent, AnalystAgent
from researcher.utils import (
    save_markdown,
    log_stage,
    get_artifact_path,
    load_artifact_from_file,
    get_llm_config,
    get_default_config_path,
    save_agent_history,
    save_json,
    load_json,
    raise_if_run_cancel_requested,
    iterable_group_chat
)
from researcher.prompts.templates import (
    RESULT_ANALYSIS_PROMPT,
    ENGINEER_STEP_PROMPT,
    EXPERIMENT_EXECUTION_CONTEXT_PROMPT,
    ENGINEER_STEP_GUIDANCE_SHORT_PROMPT,
    RA_STEP_PROMPT,
    CODE_DEBUG_PROMPT,
    INSTRUCTION_ENGINEER_PROMPT,
    VALIDATOR_PROMPT,
    REPAIR_ENGINEER_PROMPT,
)
from researcher.exceptions import WorkflowError

from researcher.integrations.opencode import (
    list_opencode_model_selectors,
    opencode_codebase_experiment,
    resolve_opencode_model_selector,
)


# =============================================================================
# Three-Role Agent Creation Functions
# Based on ENGINEER_SYSTEM_PROMPT's three responsibilities
# =============================================================================

def create_instruction_engineer_agent(llm_config):
    """Create Instruction Engineer agent for generating execution specifications."""
    return ConversableAgent(
        name="InstructionEngineer",
        system_message=INSTRUCTION_ENGINEER_PROMPT,
        llm_config=llm_config,
    )


def create_validator_agent(llm_config):
    """Create Validator agent for determining step completion."""
    return ConversableAgent(
        name="Validator",
        system_message=VALIDATOR_PROMPT,
        llm_config=llm_config,
    )


def create_repair_engineer_agent(llm_config):
    """Create Repair Engineer agent for generating targeted fixes."""
    return ConversableAgent(
        name="RepairEngineer",
        system_message=REPAIR_ENGINEER_PROMPT,
        llm_config=llm_config,
    )


def experiment_execution_node(state: ResearchState) -> Dict[str, Any]:
    workspace_dir = state["workspace_dir"]
    raise_if_run_cancel_requested(state)
    config = state["config"]["researcher"]["experiment_execution"]
    backend = config["backend"]
    max_retries = config["code_execution_retries"]
    human_in_the_loop = config["human_in_the_loop"]
    env_path = Path(config["virtual_env_path"])
    start_step = int(config.get("start_step", 1) or 1)
    resume_experiment_dir = config.get("resume_experiment_dir")
    opencode_runtime = {}
    managed_opencode_runtime = None

    try:
        task_content = load_artifact_from_file(workspace_dir, "task")
        idea_content = load_artifact_from_file(workspace_dir, "idea")
        method_content = load_artifact_from_file(workspace_dir, "method")

        steps, execution_order = parse_method_markdown(method_content)
        steps_dict = {s.step_id: s for s in steps}
        if not execution_order:
            raise WorkflowError("No execution steps parsed from method markdown.")

        llm_config = get_llm_config()

        exp_dir, resumed = _select_experiment_dir(
            workspace_dir=workspace_dir,
            start_step=start_step,
            resume_experiment_dir=resume_experiment_dir,
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_stage(
            workspace_dir,
            "experiment_execution",
            f"Using experiment dir: {exp_dir} (resumed={resumed}, start_step={start_step})",
        )

        if backend == "opencode":
            repo_root = Path(__file__).resolve().parents[2]
            # opencode_config = config.get("opencode") or {}
            # model_selector = opencode_config.get("model_selector")
            # provider_id, model_id = resolve_opencode_model_selector(model_selector)
            managed_opencode_runtime = _ManagedOpenCodeRuntime(
                workspace_dir=exp_dir,
                config_path=repo_root / "configs" / "opencode.json",
            )
            runtime_info = managed_opencode_runtime.start()
            opencode_runtime = {
                "base_url": runtime_info["base_url"],
                # "provider_id": provider_id,
                # "model_id": model_id,
                "pid": runtime_info["pid"],
                "log_path": runtime_info["log_path"],
            }
            log_stage(
                workspace_dir,
                "experiment_execution",
                f"Managed OpenCode runtime started at {opencode_runtime['base_url']} (pid={opencode_runtime['pid']})",
            )
            print(f"opencode_runtime: {opencode_runtime}")

        # Load step records for resume
        step_records = _load_step_records(exp_dir)
        step_results = _build_step_results_from_records(
            step_records=step_records,
            execution_order=execution_order,
            start_step=start_step,
        )

        # Agents
        ra = RAAgent().create_agent(llm_config)
        engineer = EngineerAgent().create_agent(llm_config)
        analyst = AnalystAgent().create_agent(llm_config)

        # Three-Role Engineer Agents (based on ENGINEER_SYSTEM_PROMPT responsibilities)
        instruction_engineer = create_instruction_engineer_agent(llm_config)
        validator = create_validator_agent(llm_config)
        repair_engineer = create_repair_engineer_agent(llm_config)

        if human_in_the_loop:
            human = ConversableAgent(name="Human", human_input_mode="ALWAYS")

        # ===============================
        # Context
        # ===============================

        ctx = ContextVariables(data={
            "execution_order": execution_order,
            "current_index": _get_start_index(execution_order, start_step),
            "current_step_id": start_step, # start from specified step
            "step_results": step_results,
            "step_status": "plan",   # plan → check → done
            "backend": backend,
            "retry_count": 0,
            "max_retries": max_retries,
            "workspace_dir": workspace_dir,
            "experiment_dir": exp_dir,
            "step_records_dir": exp_dir / "steps",
            "env_path": env_path,
            "session_id": None,
            "opencode_base_url": opencode_runtime.get("base_url"),
            # "opencode_provider_id": opencode_runtime.get("provider_id"),
            # "opencode_model_id": opencode_runtime.get("model_id"),
            # Three-role workflow state
            "workflow_phase": "instruction",  # instruction → validation → repair (loop)
            "instruction_plan": None,         # Instruction Engineer output
            "execution_result": None,         # Code execution output
            "validation_result": None,        # Validator output
            "repair_count": 0,                # Repair attempts counter
        })

        # ===============================
        # STEP DISPATCH (Start New Step)
        # ===============================

        def dispatch_next_step(ctx: ContextVariables) -> FunctionTargetResult:
            """Start a new step: reset workflow state and call Instruction Engineer."""
            index = ctx["current_index"]

            if index >= len(ctx["execution_order"]):
                return FunctionTargetResult(
                    target=AgentTarget(analyst),
                    messages="All steps finished. Analyze results.",
                    context_variables=ctx,
                )

            step_id = ctx["execution_order"][index]
            step = steps_dict[step_id]

            ctx["current_step_id"] = step_id
            step_dir = get_step_dir(step, ctx["experiment_dir"])
            ctx["step_dir"] = step_dir

            # Reset workflow state for new step
            ctx["workflow_phase"] = "instruction"
            ctx["step_status"] = "plan"
            ctx["retry_count"] = 0
            ctx["repair_count"] = 0
            ctx["step_file_snapshot"] = _get_file_snapshot(step_dir)

            prompt = parse_method_step_to_prompt(
                step=step,
                step_results=ctx["step_results"],
                exp_dir=ctx["experiment_dir"],
            )

            return FunctionTargetResult(
                target=AgentTarget(instruction_engineer),
                messages=prompt,
                context_variables=ctx,
            )

        # ===============================
        # THREE-ROLE WORKFLOW ROUTER
        # ===============================

        def route_three_role_workflow(ctx: ContextVariables) -> FunctionTargetResult:
            """Route within three-role workflow: validation → repair → validation loop."""
            step_id = ctx["current_step_id"]
            step = steps_dict[step_id]
            phase = ctx.get("workflow_phase", "validation")

            if phase == "validation":
                # Validator determines if step is complete
                execution_result = ctx.get("execution_result", "")
                expected_output = step.expected_output

                validation_prompt = (
                    f"Step {step_id}: {step.description}\n\n"
                    f"Expected Output: {expected_output}\n\n"
                    f"Execution Result:\n{execution_result}\n\n"
                    "Evaluate whether the step objective has been fully satisfied. "
                    "Provide Evidence, Gap analysis, Next action, and Completion decision."
                )

                return FunctionTargetResult(
                    target=AgentTarget(validator),
                    messages=validation_prompt,
                    context_variables=ctx,
                )

            if phase == "repair":
                # Repair Engineer generates targeted fix instructions
                validation_feedback = ctx.get("validation_result", "")
                instruction_plan = ctx.get("instruction_plan", "")

                repair_prompt = (
                    f"Original Plan:\n{instruction_plan}\n\n"
                    f"Validation Feedback:\n{validation_feedback}\n\n"
                    "Generate targeted repair instructions to fix the specific issues. "
                    "Be minimal and specific - do not rewrite the entire solution."
                )

                return FunctionTargetResult(
                    target=AgentTarget(repair_engineer),
                    messages=repair_prompt,
                    context_variables=ctx,
                )

            # Unknown phase, default to validation
            ctx["workflow_phase"] = "validation"
            return route_three_role_workflow(ctx)

        # ===============================
        # BACKEND EXECUTION
        # ===============================

        def execute_backend(instruction: str, ctx: ContextVariables):
            backend = ctx["backend"]

            if backend == "opencode":
                results, session_id = opencode_codebase_experiment(
                    instruction,
                    workspace_dir=ctx["experiment_dir"] / f"step_{ctx['current_step_id']}",
                    env_path=ctx["env_path"],
                    session_id=ctx["session_id"],
                    opencode_base_url=ctx["opencode_base_url"],
                    # provider_id=ctx["opencode_provider_id"],
                    # model_id=ctx["opencode_model_id"],
                )
                ctx["session_id"] = session_id
                #print(f"Session ID: {session_id}, returned")
                return results

            raise ValueError(f"Unsupported backend: {backend}")

        # ===============================
        # THREE-ROLE ENGINEER HANDLERS
        # Based on ENGINEER_SYSTEM_PROMPT's three responsibilities
        # ===============================

        def handle_instruction_engineer(output, ctx: ContextVariables) -> FunctionTargetResult:
            """Handle Instruction Engineer output: save plan, execute code, then validate."""
            step_id = ctx["current_step_id"]

            # Save the instruction plan
            ctx["instruction_plan"] = str(output)
            
            # Execute the instruction via backend (opencode)
            results = execute_backend(str(output), ctx)
            ctx["execution_result"] = results
            
            # Move to validation phase and route to validator
            ctx["workflow_phase"] = "validation"

            return route_three_role_workflow(ctx)

        def handle_validator(output, ctx: ContextVariables) -> FunctionTargetResult:
            """Handle Validator output: check COMPLETE/INCOMPLETE, route accordingly."""
            step_id = ctx["current_step_id"]
            step = steps_dict[step_id]
            text = str(output)

            # Save validation result
            ctx["validation_result"] = text

            # Check if COMPLETE - look for explicit completion markers
            # Must contain "==STEP_COMPLETE==" OR "Completion decision: COMPLETE" without INCOMPLETE
            is_complete = False
            if "==STEP_COMPLETE==" in text:
                is_complete = True
            elif "Completion decision: COMPLETE" in text and "INCOMPLETE" not in text:
                is_complete = True
            elif "COMPLETE" in text and "INCOMPLETE" not in text:
                # Fallback: has COMPLETE but no INCOMPLETE anywhere
                is_complete = True

            if is_complete:
                # Step is complete
                summary = _extract_step_summary(text)
                files = _collect_step_files(ctx)
                ctx["step_results"][step_id] = {
                    "status": "success",
                    "output": summary,
                    "files": files,
                }
                _save_step_record(ctx, step_id, step, status="success", summary=summary, output=text, files=files)

                # Move to next step, reset workflow state
                ctx["current_index"] += 1
                ctx["workflow_phase"] = "instruction"
                ctx["retry_count"] = 0
                ctx["repair_count"] = 0
                ctx["instruction_plan"] = None
                ctx["execution_result"] = None
                ctx["validation_result"] = None

                return dispatch_next_step(ctx)

            # Step is INCOMPLETE - need repair
            ctx["repair_count"] += 1

            # Check if max retries exceeded
            if ctx["repair_count"] >= ctx["max_retries"]:
                summary = _extract_step_summary(text)
                files = _collect_step_files(ctx)
                ctx["step_results"][step_id] = {
                    "status": "failed",
                    "output": summary,
                    "files": files,
                }
                _save_step_record(ctx, step_id, step, status="failed", summary=summary, output=text, files=files)

                # Move to next step even though this one failed
                ctx["current_index"] += 1
                ctx["workflow_phase"] = "instruction"
                ctx["retry_count"] = 0
                ctx["repair_count"] = 0

                return dispatch_next_step(ctx)

            # Move to repair phase and route to repair engineer
            ctx["workflow_phase"] = "repair"
            return route_three_role_workflow(ctx)

        def handle_repair_engineer(output, ctx: ContextVariables) -> FunctionTargetResult:
            """Handle Repair Engineer output: merge with original plan, re-execute, then re-validate."""
            repair_instructions = str(output)

            # Merge repair instructions with original plan
            original_plan = ctx.get("instruction_plan", "")
            merged_plan = (
                f"REPAIR INSTRUCTIONS:\n{repair_instructions}\n\n"
                f"ORIGINAL PLAN:\n{original_plan}"
            )

            # Re-execute the merged plan
            ctx["instruction_plan"] = merged_plan
            results = execute_backend(merged_plan, ctx)
            ctx["execution_result"] = results
            
            # Move back to validation phase and route to validator
            ctx["workflow_phase"] = "validation"

            return route_three_role_workflow(ctx)

        # ===============================
        # LEGACY HANDLER (kept for RA tasks)
        # ===============================

        def handle_engineer(output, ctx: ContextVariables) -> FunctionTargetResult:
            raise RuntimeError("Invalid step_status")

        # ===============================
        # RA HANDLER
        # ===============================

        def handle_ra(output, ctx: ContextVariables) -> FunctionTargetResult:
            step_id = ctx["current_step_id"]
            step = steps_dict[step_id]
            summary = str(output)
            files = _collect_step_files(ctx)

            ctx["step_results"][step_id] = {
                "status": "success",
                "output": summary,
                "files": files,
            }
            _save_step_record(ctx, step_id, step, status="success", summary=summary, output=summary, files=files)

            ctx["current_index"] += 1
            return dispatch_next_step(ctx)

        # ===============================
        # AGENT ROUTER (Three-Role Workflow)
        # ===============================

        def route_output(output, ctx: ContextVariables):
            step_id = ctx.get("current_step_id")
            step = steps_dict.get(step_id)

            if not step:
                return FunctionTargetResult(
                    target=TerminateTarget(),
                    messages=None,
                    context_variables=ctx,
                )

            # RA tasks use legacy handler
            if step.assignee == "RA":
                return handle_ra(output, ctx)

            # Engineer tasks use three-role workflow
            phase = ctx.get("workflow_phase", "instruction")

            if phase == "instruction":
                return handle_instruction_engineer(output, ctx)
            if phase == "validation":
                return handle_validator(output, ctx)
            if phase == "repair":
                return handle_repair_engineer(output, ctx)
            # Fallback to validation if phase is unknown
            return handle_validator(output, ctx)

        # Register handoffs for all agents
        ra.handoffs.set_after_work(FunctionTarget(lambda output, context_variables:
                                                  route_output(output, context_variables)))
        engineer.handoffs.set_after_work(FunctionTarget(lambda output, context_variables:
                                                        route_output(output, context_variables)))
        
        # Three-role engineer agents use the same router
        instruction_engineer.handoffs.set_after_work(FunctionTarget(lambda output, context_variables:
                                                                    route_output(output, context_variables)))
        validator.handoffs.set_after_work(FunctionTarget(lambda output, context_variables:
                                                         route_output(output, context_variables)))
        repair_engineer.handoffs.set_after_work(FunctionTarget(lambda output, context_variables:
                                                               route_output(output, context_variables)))
        analyst.handoffs.set_after_work(TerminateTarget())

        # ===============================
        # PATTERN
        # ===============================

        # Determine initial agent based on first step assignee
        first_step = steps_dict[execution_order[ctx["current_index"]]]
        if first_step.assignee == "RA":
            initial_agent = ra
        else:
            initial_agent = instruction_engineer  # Start with instruction phase for Engineer tasks

        pattern = DefaultPattern(
            initial_agent=initial_agent,
            agents=[ra, engineer, analyst, instruction_engineer, validator, repair_engineer],
            context_variables=ctx,
            group_manager_args={"llm_config": llm_config},
        )


        # Construct the inital messages
        # 1. Research Context
        context_prompt = EXPERIMENT_EXECUTION_CONTEXT_PROMPT.format(
            task=task_content,
            idea=extract_selected_idea(idea_content),
        )
        #context_messages = TextMessage(content=context_prompt, source="user")
        # 2. First Step Prompt
        first_step = steps_dict[execution_order[ctx["current_index"]]]
        first_step_prompt = parse_method_step_to_prompt(first_step, step_results=ctx["step_results"], exp_dir=exp_dir)
        #first_step_messages = TextMessage(content=first_step_prompt, source="user")

        initial_messages = [
            {"role": "user", "content": context_prompt + first_step_prompt},
            #{"role": "user", "content": first_step_prompt}
        ]
        if state["config"]["researcher"]["iterable"]:
            result, context, last_agent = iterable_group_chat(
                state,
                max_rounds=len(execution_order) * (max_retries + 2) + 5,
                enable_hitl=False,
                pattern=pattern,
                prompt=initial_messages,
            )
        else:
            result, context, _ = initiate_group_chat(
                pattern=pattern,
                messages=initial_messages,
                max_rounds=len(execution_order) * (max_retries + 2) + 5,
            )

        # Extract analysis from analyst messages (no minimum length requirement).
        analyst_messages = [
            msg.get("content")
            for msg in result.chat_history
            if msg.get("content")
        ]
        analysis = analyst_messages[-1] if analyst_messages else None

        # if not analysis:
        #     error_msg = "[✕] No analysis message found from analyst agent."
        #     log_stage(workspace_dir, "experiment_execution", error_msg)
        #     raise WorkflowError("[✕]Analyst analysis is required but not found. Check agent execution flow.")

        all_result_files = []
        for step_id in execution_order:
            step_result = context.get("step_results", {}).get(step_id, {})
            step_files = step_result.get("files", [])
            all_result_files.extend([Path(workspace_dir / f) for f in step_files if Path(workspace_dir / f).exists()])

        all_result_files = list(set(all_result_files))

        figure_extensions = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".gif", ".bmp", ".tiff"}
        data_files = [p for p in all_result_files if p.suffix.lower() not in figure_extensions]
        figure_files = [p for p in all_result_files if p.suffix.lower() in figure_extensions]

        log_stage(
            workspace_dir,
            "experiment_execution",
            f"Final file collection: {len(data_files)} data files, {len(figure_files)} figure files",
        )

        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="experiment_execution",
            messages=result.chat_history,
            agent_chat_messages={
                ra.name: ra.chat_messages,
                engineer.name: engineer.chat_messages,
                #code_debugger.name: code_debugger.chat_messages,
                analyst.name: analyst.chat_messages,
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

        log_stage(
            workspace_dir,
            "experiment_execution",
            f"Completed. Generated {len(data_files)} data files, {len(figure_files)} figure files",
        )

        update_state = {
            "results": exp_result,
            "stage": "experiment_execution",
            "opencode": {
                "base_url": opencode_runtime.get("base_url"),
                # "provider_id": opencode_runtime.get("provider_id"),
                # "model_id": opencode_runtime.get("model_id"),
                "session_id": context.get("session_id"),
            } if backend == "opencode" else None,
        }
        # router
        if state["config"]["researcher"]["workflow"] == "default":
            update_state["next_node"] = "report_generation"
        return update_state
    
    # Ensure that unexpected exits or interruptions do not result in the leakage of residual subprocess.
    finally:
        if managed_opencode_runtime is not None:
            managed_opencode_runtime.stop()
            log_stage(
                workspace_dir,
                "experiment_execution",
                "Managed OpenCode runtime stopped",
            )


class _ManagedOpenCodeRuntime:
    def __init__(self, workspace_dir: Path, config_path: Path):
        self.workspace_dir = workspace_dir
        self.config_path = config_path
        self.host = "127.0.0.1"
        # Create a temporary TCP socket and bind to port 0 so the OS picks a free port.
        # - `port=0`: request an ephemeral port from the kernel
        # - `getsockname()[1]`: read the assigned port from the socket address tuple
        # Note: the socket is closed when the `with` block ends; we only use it to reserve
        # an available port number before the real server is started in `start()`.
        #
        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        #     sock.bind((self.host, 0))
        #     # Allow quick reuse of the address/port to reduce TIME_WAIT binding issues.
        #     sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #     self.port = int(sock.getsockname()[1])
        self.port = 4096
        # Base URL for subsequent clients (i.e., the OpenCode server).
        self.base_url = f"http://{self.host}:{self.port}"
        self.log_path = workspace_dir / "opencode_runtime.log"
        self.meta_path = workspace_dir / "opencode_runtime.json"
        self.process: Optional[subprocess.Popen] = None
        self._stopped = False
        self._registered = False

    def start(self) -> Dict[str, Any]:
        self._cleanup_stale_runtime()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self.log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["OPENCODE_CONFIG"] = str(self.config_path)
        env["OPENCODE_HOST"] = self.host
        env["OPENCODE_PORT"] = str(self.port)
        env["OPENCODE_MANAGED"] = "1"

        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "start_opencode.sh"
        self.process = subprocess.Popen(
            ["bash", str(script_path)],
            cwd=str(repo_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # Start a new process group to manage the server and its subprocesses.
            text=True,
        )
        # Register the stop method to be called when the program exits.
        atexit.register(self.stop)
        self._registered = True
        save_json(
            {
                "pid": self.process.pid,
                "host": self.host,
                "port": self.port,
                "base_url": self.base_url,
                "log_path": str(self.log_path),
                "config_path": str(self.config_path),
                "started_at": time.time(),
            },
            self.meta_path,
        )
        try:
            deadline = time.time() + 30
            connected = False
            while time.time() < deadline:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.5)
                    try:
                        sock.connect((self.host, self.port))
                        connected = True
                        break
                    except OSError:
                        time.sleep(0.5)
            if not connected:
                raise WorkflowError(f"Timed out waiting for OpenCode server at {self.host}:{self.port}")
        except Exception:
            self.stop()
            tail = ""
            if self.log_path.exists():
                try:
                    lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    tail = "\n".join(lines[-40:])
                except OSError:
                    tail = ""
            detail = f"\nOpenCode log tail:\n{tail}" if tail else ""
            raise WorkflowError(f"Failed to start managed OpenCode runtime.{detail}")
        return {
            "base_url": self.base_url,
            "pid": self.process.pid,
            "log_path": str(self.log_path),
        }

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._registered:
            try:
                atexit.unregister(self.stop)
            except Exception:
                pass
            self._registered = False
        proc = self.process
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self.meta_path.exists():
            try:
                self.meta_path.unlink()
            except OSError:
                pass

    def _cleanup_stale_runtime(self) -> None:
        if not self.meta_path.exists():
            return
        try:
            meta = load_json(self.meta_path)
        except Exception:
            meta = None
        if not isinstance(meta, dict):
            try:
                self.meta_path.unlink()
            except OSError:
                pass
            return
        pid = meta.get("pid")
        if isinstance(pid, int):
            try:
                os.killpg(pid, signal.SIGTERM)
                time.sleep(1)
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                pass
        try:
            self.meta_path.unlink()
        except OSError:
            pass


# utils

def parse_method_step_to_prompt(step: MethodStep, step_results: dict, exp_dir: Path) -> str:
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
    
    if step.assignee == "RA":
        prompt = RA_STEP_PROMPT.format(
            step_id=step.step_id,
            description=step.description,
            expected_output=step.expected_output,
            context=context_str,
            available_files=files_str
        )
    else:  # Engineer
        timestamp = exp_dir.name.replace("code_", "")
        step_dir = exp_dir / f"step_{step.step_id}"
        prompt = ENGINEER_STEP_GUIDANCE_SHORT_PROMPT.format(
            step_id=step.step_id,
            description=step.description,
            expected_output=step.expected_output,
            available_files=files_str,
            context=context_str,
            step_dir=step_dir,
            timestamp=timestamp
        )
    return prompt



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

def get_step_dir(step: MethodStep, exp_dir: Path) -> Path:
    """Get per-step experiment directory"""
    step_dir = exp_dir / f"step_{step.step_id}"
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


def _select_experiment_dir(workspace_dir: Path, start_step: int, resume_experiment_dir: Optional[str]) -> tuple[Path, bool]:
    exp_root = workspace_dir / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)

    if resume_experiment_dir:
        p = Path(resume_experiment_dir)
        if not p.is_absolute():
            p = workspace_dir / resume_experiment_dir
        if p.exists():
            return p, True

    if start_step > 1:
        candidates = [d for d in exp_root.iterdir() if d.is_dir() and d.name.startswith("exp_")]
        candidates.sort()
        if candidates:
            return candidates[-1], True

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return exp_root / f"exp_{timestamp}", False


def _load_step_records(exp_dir: Path) -> dict:
    records_dir = exp_dir / "steps"
    if not records_dir.exists():
        return {}
    records = {}
    for path in records_dir.glob("step_*.json"):
        data = load_json(path)
        if not data:
            continue
        step_id = data.get("step_id")
        if isinstance(step_id, int):
            records[step_id] = data
    return records


def _build_step_results_from_records(step_records: dict, execution_order: List[int], start_step: int) -> dict:
    if not step_records:
        return {}
    allowed = set()
    for sid in execution_order:
        if sid < start_step:
            allowed.add(sid)
    results = {}
    for sid, record in step_records.items():
        if sid not in allowed:
            continue
        summary = record.get("summary") or record.get("output") or ""
        results[sid] = {
            "status": record.get("status", "unknown"),
            "output": summary,
            "files": record.get("files", []),
        }
    return results


def _get_start_index(execution_order: List[int], start_step: int) -> int:
    try:
        return execution_order.index(start_step)
    except ValueError:
        return 0


def _extract_step_summary(text: str) -> str:
    if "==STEP_COMPLETE==" not in text:
        return text.strip()
    # Remove completion marker and anything after it
    parts = text.split("==STEP_COMPLETE==", 1)
    summary = parts[0].strip()
    return summary or "Step completed"


def _collect_step_files(ctx: ContextVariables) -> list[str]:
    step_dir = ctx.get("step_dir")
    workspace_dir = ctx.get("workspace_dir")
    if not step_dir or not workspace_dir:
        return []
    before = ctx.get("step_file_snapshot") or set()
    after = _get_file_snapshot(step_dir)
    new_files = sorted(after - before)
    if not new_files:
        new_files = sorted(after)
    rel_files = []
    for p in new_files:
        try:
            rel_files.append(str(p.relative_to(workspace_dir)))
        except Exception:
            rel_files.append(str(p))
    return rel_files


def _save_step_record(
    ctx: ContextVariables,
    step_id: int,
    step: MethodStep,
    status: str,
    summary: str,
    output: str,
    files: list[str],
) -> None:
    records_dir = ctx.get("step_records_dir")
    if not records_dir:
        return
    try:
        from datetime import datetime
        record = {
            "step_id": step_id,
            "status": status,
            "summary": summary,
            "output": output,
            "files": files,
            "assignee": step.assignee,
            "dependencies": step.dependencies,
            "expected_output": step.expected_output,
            "step_dir": str(ctx.get("step_dir", "")),
            "completed_at": datetime.now().isoformat(),
        }
        save_json(record, Path(records_dir) / f"step_{step_id}.json")
    except Exception:
        # Best-effort only
        pass
