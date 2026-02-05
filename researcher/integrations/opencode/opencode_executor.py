"""
OpenCode executor. Usage:

  from researcher.integrations.opencode import opencode_codebase_experiment
  result = await opencode_codebase_experiment("instruction", workspace_dir)  # -> {text, parts, id, session_id, role, time, parent_id, model_id, provider_id, mode, agent, path, cost, tokens, finish, raw}
"""

from pathlib import Path
from typing import Dict, Any, Optional

try:
    from researcher.integrations.opencode import opencode_client

    OpenCodeClient = opencode_client.OpenCodeClient
    check_opencode_availability = opencode_client.check_opencode_availability
    OPENCODE_AVAILABLE = check_opencode_availability()
except ImportError as e:
    raise ImportError(f"OpenCode client import failed: {e}")


OPENCODE_NOTE = """## Workflow
- After writing or changing code, run it automatically. Use the run result to decide whether to fix or extend the program; repeat until the instruction is fully satisfied.
- Prefer **edit** for small changes; use **read** before editing. Then summarize the experiment result so that reading the summary alone is enough to understand the result of this step."""




class OpenCodeExecutor:
    """instruction → execution → result cycle"""

    def __init__(
        self,
        opencode_base_url: str = "http://localhost:4096",
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

    def execute_instruction(self, instruction: str, workspace_dir: Path) -> Dict[str, Any]:
        # Prepend debugging guidance to instruction
        full_instruction = OPENCODE_NOTE + "\n\n" + instruction

        with OpenCodeClient(
            base_url=self.opencode_base_url,
            timeout=self.timeout,
            config_file=self.config_file,
            provider_id=self.provider_id,
            model_id=self.model_id,
        ) as opencode:
            opencode.create_session(workspace_dir)

            # with auto-debugging
            #print("Sending instruction to OpenCode:", full_instruction)
            raw = opencode.send_instruction(full_instruction)
            #print("OpenCode result:", raw)
            out = self._parse_response(raw)
            #out["raw"] = raw
            return out
        
    def _parse_response(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        #info = raw.get("info") or raw
        parts = raw.get("parts") or []
        text_response = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        return text_response


_default_executor: Optional[OpenCodeExecutor] = None


def _get_default_executor() -> OpenCodeExecutor:
    global _default_executor
    if _default_executor is None:
        _default_executor = OpenCodeExecutor()
    return _default_executor


def opencode_codebase_experiment(instruction: str, workspace_dir: Path) -> Dict[str, Any]:
    text_response =  _get_default_executor().execute_instruction(instruction, workspace_dir)

    # Parse
    # TODO
    exp_results = text_response


    return exp_results



if __name__ == "__main__":
    def _main():
        return opencode_codebase_experiment(
            "Write a simple Python script that prints 'Hello, OpenCode!'",
            Path("."),
        )

    print(_main())