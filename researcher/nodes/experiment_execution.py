from typing import Dict, Any, List, Optional
from pathlib import Path
import re

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
    deduplicate_long_repeats,
    save_json,
    load_json,
    iterable_group_chat
)
from researcher.prompts.templates import (
    RESULT_ANALYSIS_PROMPT,
    ENGINEER_STEP_PROMPT,
    EXPERIMENT_EXECUTION_CONTEXT_PROMPT,
    ENGINEER_STEP_GUIDANCE_SHORT_PROMPT,
    RA_STEP_PROMPT,
    CODE_DEBUG_PROMPT,
)
from researcher.exceptions import WorkflowError

from researcher.integrations.opencode import (
    get_opencode_server_url,
    list_opencode_model_selectors,
    opencode_codebase_experiment,
    resolve_opencode_model_selector,
)

def experiment_execution_node(state: ResearchState) -> Dict[str, Any]:
    workspace_dir = state["workspace_dir"]
    config = state["config"]["researcher"]["experiment_execution"]
    backend = config["backend"]
    max_retries = config["code_execution_retries"]
    human_in_the_loop = config["human_in_the_loop"]
    env_path = Path(config["virtual_env_path"])
    start_step = int(config.get("start_step", 1) or 1)
    resume_experiment_dir = config.get("resume_experiment_dir")
    opencode_runtime = {}

    if backend == "opencode":
        opencode_config = config.get("opencode") or {}
        model_selector = opencode_config.get("model_selector")
        provider_id, model_id = resolve_opencode_model_selector(model_selector)
        opencode_runtime = {
            "base_url": opencode_config.get("base_url") or get_opencode_server_url(),
            "provider_id": provider_id,
            "model_id": model_id,
        }
        print(f"opencode_runtime: {opencode_runtime}")
        

    task_content = load_artifact_from_file(workspace_dir, "task")
    idea_content = load_artifact_from_file(workspace_dir, "idea")
    method_content = load_artifact_from_file(workspace_dir, "method")

    steps, execution_order = parse_method_markdown(method_content)
    steps_dict = {s.step_id: s for s in steps}

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
        "opencode_provider_id": opencode_runtime.get("provider_id"),
        "opencode_model_id": opencode_runtime.get("model_id"),
    })

    # ===============================
    # STEP DISPATCH
    # ===============================

    def dispatch_next_step(ctx: ContextVariables) -> FunctionTargetResult:
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
        ctx["step_status"] = "plan"
        ctx["retry_count"] = 0
        step_dir = get_step_dir(step, ctx["experiment_dir"])
        ctx["step_dir"] = step_dir
        ctx["step_file_snapshot"] = _get_file_snapshot(step_dir)

        prompt = parse_method_step_to_prompt(
            step=step,
            step_results=ctx["step_results"],
            exp_dir=ctx["experiment_dir"],
        )

        target_agent = ra if step.assignee == "RA" else engineer

        return FunctionTargetResult(
            target=AgentTarget(target_agent),
            messages=prompt,
            context_variables=ctx,
        )

    # ===============================
    # BACKEND EXECUTION
    # ===============================

    def execute_backend(instruction: str, ctx: ContextVariables):
        backend = ctx["backend"]

        if backend == "opencode":
            results, session_id = opencode_codebase_experiment(
                instruction,
                workspace_dir=ctx["experiment_dir"] / f"step_{ctx["current_step_id"]}",
                env_path=ctx["env_path"],
                session_id=ctx["session_id"],
                opencode_base_url=ctx["opencode_base_url"],
                provider_id=ctx["opencode_provider_id"],
                model_id=ctx["opencode_model_id"],
            )
            ctx["session_id"] = session_id
            #print(f"Session ID: {session_id}, returned")
            return results
        
        raise ValueError(f"Unsupported backend: {backend}")

    # ===============================
    # ENGINEER HANDLER
    # ===============================

    def handle_engineer(output, ctx: ContextVariables) -> FunctionTargetResult:
        step_id = ctx["current_step_id"]
        status = ctx["step_status"]
        step = steps_dict[step_id]

        # ---- PLAN → EXECUTE ----
        if status == "plan":
            results = execute_backend(str(output), ctx)
            ctx["last_execution_output"] = results
            ctx["step_status"] = "check"

            return FunctionTargetResult(
                target=AgentTarget(engineer),
                messages=(
                    f"Execution Results:\n{results}\n\n"

                    "Evaluate whether this step objective has been fully satisfied based on the execution evidence.\n"
                    "Do NOT assume success unless it is supported by concrete artifacts, files, or metrics.\n\n"

                    "Do NOT mark completion for:\n"
                    "- partial progress\n"
                    "- demo-only output\n"
                    "- smoke-test-only output\n"
                    "- empty or placeholder artifacts\n"
                    "- outputs that do not satisfy the step objective\n\n"

                    "A step is COMPLETE only if ALL of the following are true:\n"
                    "1) The required deliverable or artifact for this step has been produced.\n"
                    "2) The artifact/output is correct and satisfies the intended objective (not just runnable).\n"
                    "3) There is concrete evidence from execution outputs, logs, files, or metrics.\n"
                    "4) No blocking runtime errors remain.\n\n"

                    "If any condition above is not satisfied, the step must be marked INCOMPLETE.\n"
                    "Diagnose the issue and propose the minimal next action required to fix it.\n\n"

                    "Your response must include:\n"
                    "- Evidence: key artifacts, files, logs, or metrics proving correctness\n"
                    "- Gap analysis: what is missing, incorrect, or failing (if any)\n"
                    "- Next action: the minimal concrete fix or verification that should run next\n\n"
                    "- Completion decision: your final decision after analysis, COMPLETE or INCOMPLETE\n"

                    "Prefer targeted fixes such as:\n"
                    "- correcting file paths\n"
                    "- fixing dataset loading\n"
                    "- adjusting configuration parameters\n"
                    "- resolving integration bugs\n\n"

                    "Avoid restarting the entire step unless absolutely necessary.\n\n"

                    "Append '==STEP_COMPLETE==' ONLY when the completion decision is COMPLETE."
                ),
                context_variables=ctx,
            )

        # ---- CHECK COMPLETION ----
        if status == "check":
            text = str(output)

            if "==STEP_COMPLETE==" in text:
                summary = _extract_step_summary(text)
                files = _collect_step_files(ctx)
                ctx["step_results"][step_id] = {
                    "status": "success",
                    "output": summary,
                    "files": files,
                }
                _save_step_record(ctx, step_id, step, status="success", summary=summary, output=text, files=files)
                ctx["current_index"] += 1
                ctx["step_status"] = "done"
                return dispatch_next_step(ctx)

            # retry
            ctx["retry_count"] += 1

            if ctx["retry_count"] >= ctx["max_retries"]:
                summary = _extract_step_summary(text)
                files = _collect_step_files(ctx)
                ctx["step_results"][step_id] = {
                    "status": "failed",
                    "output": summary,
                    "files": files,
                }
                _save_step_record(ctx, step_id, step, status="failed", summary=summary, output=text, files=files)
                ctx["current_index"] += 1
                return dispatch_next_step(ctx)

            # re-plan
            ctx["step_status"] = "plan"

            return FunctionTargetResult(
                target=AgentTarget(engineer),
                messages=(
                    "Step is incomplete. Analyze the gaps described above and issue specific corrective instructions to resolve them. The fix plan must target the exact failure points (e.g., missing files, invalid metrics, dependency errors, or incorrect data processing) and define the minimal actions required to repair the step and rerun it. Do not mark the step as complete until the required outputs exist and are verified by concrete artifacts, logs, or metrics."
                ),
                context_variables=ctx,
            )

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
    # AGENT ROUTER
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

        if step.assignee.lower() == "engineer":
            target = handle_engineer(output, ctx)

        elif step.assignee == "RA":
            target = handle_ra(output, ctx)
        return target

    # Register handoffs
    ra.handoffs.set_after_work(FunctionTarget(lambda output, context_variables: 
                                              route_output(output, context_variables)))
    engineer.handoffs.set_after_work(FunctionTarget(lambda output, context_variables: 
                                              route_output(output, context_variables)))
    analyst.handoffs.set_after_work(TerminateTarget())

    # ===============================
    # PATTERN
    # ===============================

    pattern = DefaultPattern(
        initial_agent=ra if steps_dict[execution_order[ctx["current_index"]]].assignee == "RA" else engineer,
        agents=[ra, engineer, analyst],
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

    log_stage(workspace_dir, "experiment_execution", f"Completed. Generated {len(data_files)} data files, {len(figure_files)} figure files")

    update_state = {
        "results": exp_result,
        "stage": "experiment_execution",
        "opencode": {
            "base_url": opencode_runtime.get("base_url"),
            "provider_id": opencode_runtime.get("provider_id"),
            "model_id": opencode_runtime.get("model_id"),
            "session_id": context.get("session_id"),
        } if backend == "opencode" else None,
    }
    # router
    if state["config"]["researcher"]["workflow"] == "default":
        update_state["next_node"] = "report_generation"
    return update_state
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
