"""
OpenCode Executor Module
"""

from pathlib import Path
from typing import Dict, Any, Optional

try:
    import opencode_client

    OpenCodeClient = opencode_client.OpenCodeClient
    check_opencode_availability = opencode_client.check_opencode_availability
    OPENCODE_AVAILABLE = check_opencode_availability()
except ImportError as e:
    raise ImportError(f"OpenCode client import failed: {e}")


OPENCODE_DEBUGGING_NOTE = """## Debugging (when errors occur)
- Prefer the **edit** tool to fix specific lines or sections; avoid overwriting entire files with the write tool when only a small change is needed.
- Use **read** to inspect file contents before editing. Make minimal, targeted fixes and re-run."""


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

    async def execute_instruction(self, instruction: str, workspace_dir: Path) -> Dict[str, Any]:
        # Prepend debugging guidance to instruction
        full_instruction = OPENCODE_DEBUGGING_NOTE + "\n\n" + instruction

        async with OpenCodeClient(
            base_url=self.opencode_base_url,
            timeout=self.timeout,
            config_file=self.config_file,
            provider_id=self.provider_id,
            model_id=self.model_id,
        ) as opencode:
            await opencode.create_session(workspace_dir)

            # with auto-debugging
            print("Sending instruction to OpenCode:", full_instruction)
            result = await opencode.send_instruction(full_instruction)
            print("OpenCode result:", result)

            return result

    @staticmethod
    def prepare_instruction(instruction: str) -> str:
        return OPENCODE_DEBUGGING_NOTE + "\n\n" + instruction

    @staticmethod
    def is_available() -> bool:
        return OPENCODE_AVAILABLE

async def test_executor():
    executor = OpenCodeExecutor()

    result = await executor.execute_instruction(
        "Write a simple Python script that prints 'Hello, OpenCode!'", 
        Path(".")
    )

    return result

if __name__ == "__main__":
    import asyncio
    
    print(asyncio.run(test_executor()))