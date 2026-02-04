from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    from opencode_ai import AsyncOpencode

    OPENCODE_AVAILABLE = True
except ImportError:
    OPENCODE_AVAILABLE = False


def check_opencode_availability() -> bool:
    return OPENCODE_AVAILABLE


class OpenCodeClient:
    def __init__(
        self,
        base_url: str = "http://localhost:4096",
        timeout: float = 300.0,
        config_file: Optional[Path] = None,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        if not OPENCODE_AVAILABLE:
            raise ImportError("opencode-ai not available")
        self.client = AsyncOpencode(base_url=base_url, timeout=timeout)
        self.session_id: Optional[str] = None
        self._work_dir: Optional[Path] = None
        self._provider_id = provider_id or "qwen-local"
        self._model_id = model_id or "/home/ai_researcher/.cache/modelscope/hub/models/Qwen/Qwen3-Coder-30B-A3B-Instruct"
        _ = config_file

    async def create_session(self, work_dir: Path) -> str:
        session = await self.client.session.create(extra_body={})
        self.session_id = session.id
        self._work_dir = work_dir
        return self.session_id

    async def send_instruction(self, instruction: str) -> Dict[str, Any]:
        if not self.session_id:
            raise RuntimeError("session not created")

        parts = [{"type": "text", "text": instruction}]
        tools = {
            "bash": True,
            "edit": True,
            "write": True,
            "read": True,
            "grep": True,
            "glob": True,
            "list": True,
            "patch": True,
        }

        print(f"session_id: {self.session_id}")
        print(f"model_id: {self._model_id}")
        print(f"provider_id: {self._provider_id}")
        print(f"parts: {parts}")
        print(f"tools: {tools}")

        resp = await self.client.session.chat(
            id=self.session_id,
            model_id=self._model_id,
            provider_id=self._provider_id,
            parts=parts,
            tools=tools,
        )

        print(f"resp: {resp}")
        
        return resp.model_dump()

    async def close_session(self) -> None:
        if self.session_id:
            await self.client.session.delete(self.session_id)
            self.session_id = None

    async def __aenter__(self) -> "OpenCodeClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close_session()


async def test_this():
    client = OpenCodeClient()
    await client.create_session(Path("."))

    result = await client.send_instruction(
        "Write a python script to print 'hello 12345', and report the path of the script."
    )
    await client.close_session()

    return result


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(test_this()))