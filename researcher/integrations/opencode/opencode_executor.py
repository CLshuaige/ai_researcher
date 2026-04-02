from pathlib import Path
from typing import Dict, Any, Optional
from researcher.prompts.templates import OPENCODE_NOTE

try:
    from researcher.integrations.opencode import opencode_client

    OpenCodeClient = opencode_client.OpenCodeClient
    check_opencode_availability = opencode_client.check_opencode_availability
    OPENCODE_AVAILABLE = check_opencode_availability()
except ImportError as e:
    raise ImportError(f"OpenCode client import failed: {e}")




class OpenCodeExecutor:
    """instruction → execution → result cycle"""

    def __init__(
        self,
        opencode_base_url: str = "http://0.0.0.0:4096",
        timeout: Optional[float] = None,
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
            exp_dir=str(workspace_dir),
            exp_root=str(exp_root),
            env_path=str(env_path),
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
    timeout: Optional[float] = None,
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
