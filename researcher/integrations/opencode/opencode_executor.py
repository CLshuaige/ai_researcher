from pathlib import Path
from typing import Dict, Any, Optional

try:
    from researcher.integrations.opencode import opencode_client

    OpenCodeClient = opencode_client.OpenCodeClient
    check_opencode_availability = opencode_client.check_opencode_availability
    OPENCODE_AVAILABLE = check_opencode_availability()
except ImportError as e:
    raise ImportError(f"OpenCode client import failed: {e}")


OPENCODE_NOTE = """## Instructions

### Core Integrity Rules
- It is prohibited to fabricate or occupy various data, processes or results.
- Never invent experimental results, dataset values, file contents, logs, metrics, or execution outputs.
- If required information or data is missing, explicitly state that the information is unavailable instead of guessing.

### Experimental Discipline
- It is more important to strictly follow the experimental constraints and carry out each step meticulously to obtain the results, rather than simply completing the process.
- The objective is to obtain **real experimental outputs**, not simulated outcomes.

### Mandatory Execution Loop
- After writing or changing code, run it automatically.
- Use the real execution result to decide whether to fix or extend the program.
- If execution fails, debug and rerun until the problem is resolved.
- Repeat the following cycle until the instruction is fully satisfied:

    Write → Run → Observe Output → Verify → Fix → Re-run

- Never assume code works without executing it.

### Real Output Requirement
- All conclusions must be supported by **actual execution outputs, logs, or generated files**.
- Do not simulate outputs, mock results, or provide hypothetical execution results.
- If execution produces empty, incomplete, or invalid outputs, treat it as failure and fix the code.

### Editing Workflow
- Prefer **edit** for small changes.
- Use **read** before editing to understand the existing code context.

### Experiment Result Reporting
- After finishing this step, summarize the **actual experiment result**.
- The summary must be based only on real execution outputs.
- The summary should be detailed enough that reading it alone is sufficient to understand the result of this step.

### Execution Environment
- Use the conda environment "{env_path}" to run the code.

### Directory Scope
- Focus only on this experiment scope.
- Current step working directory: "{exp_dir}".
- Put all new files for this step inside this directory.

### Cross-Step Restrictions
- Cross-step continuity is allowed only inside this same experiment root: "{exp_root}" (for example, sibling step directories under the same root).
- Do not read, reference, or copy files from other timestamped project directories outside "{exp_root}", even if their tasks look similar.

### Completion Requirements
- Execute this experiment step to completion and produce the required step result.
- Do not stop at a partial proof-of-concept.
- A step is considered complete only if:

    1. The code has been executed successfully.
    2. The required outputs are generated.
    3. The outputs are real execution results.
    4. No placeholder implementations remain.

### Forbidden Outputs
- Avoid ending with demo scripts, toy examples, or smoke tests as final output for the step.
- In general, do not write demo code unless the instruction explicitly asks for a demo.
- The following are strictly forbidden unless explicitly requested:

    - mock data
    - placeholder implementations
    - simulated results
    - pseudo outputs
    - incomplete demo pipelines
"""




class OpenCodeExecutor:
    """instruction → execution → result cycle"""

    def __init__(
        self,
        opencode_base_url: str = "http://0.0.0.0:4096",
        timeout: float = 300.0,
        config_file: Optional[Path] = None,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ):

        if not OPENCODE_AVAILABLE:
            raise ImportError("OpenCode SDK is not available")

        self.opencode_base_url = opencode_base_url
        self.timeout = timeout
        self.config_file = config_file
        self.provider_id = provider_id
        self.model_id = model_id

    def execute_instruction(
        self,
        instruction: str,
        workspace_dir: Path,
        env_path: Path,
        session_id: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:

        exp_root = workspace_dir.parent if workspace_dir.name.startswith("step_") else workspace_dir
        full_instruction = OPENCODE_NOTE.format(
            exp_dir=workspace_dir,
            exp_root=exp_root,
            env_path=env_path,
        ) + "\n\n" + instruction

        with OpenCodeClient(
            base_url=self.opencode_base_url,
            timeout=self.timeout,
            config_file=self.config_file,
            provider_id=self.provider_id,
            model_id=self.model_id,
        ) as opencode:
            # If session_id is provided, continue that session; otherwise create a new one.
            if session_id:
                opencode.session_id = session_id
            else:
                opencode.create_session(workspace_dir)

            # with auto-debugging
            #print("Sending instruction to OpenCode:", full_instruction)
            raw = opencode.send_instruction(full_instruction)
            #print("OpenCode result:", raw)
            text = self._parse_response(raw)
            
            return text, opencode.session_id
        
    def _parse_response(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        #info = raw.get("info") or raw
        parts = raw.get("parts") or []
        text_response = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        return text_response


def opencode_codebase_experiment(
    instruction: str,
    workspace_dir: Path,
    env_path: Path,
    session_id: Optional[str] = None,
    *,
    opencode_base_url: str = "http://localhost:4096",
    timeout: float = 300.0,
    config_file: Optional[Path] = None,
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    executor = OpenCodeExecutor(
        opencode_base_url=opencode_base_url,
        timeout=timeout,
        config_file=config_file,
        provider_id=provider_id,
        model_id=model_id,
    )
    text_response, session_id = executor.execute_instruction(
        instruction, workspace_dir, env_path, session_id=session_id
    )

    exp_results = text_response

    return exp_results, session_id



if __name__ == "__main__":
    def _main():
        return opencode_codebase_experiment(
            "Write a simple Python script that prints 'Hello, OpenCode!'",
            Path("."),
            Path("."),
            session_id="ses_3d31674e3ffeI5MekbM50tgEfZ"
        )

    print(_main())
