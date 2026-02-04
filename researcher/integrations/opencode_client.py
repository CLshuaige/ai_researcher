from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

try:
    import sys

    sys.path.insert(
        0,
        "/home/ai_researcher/projects/workplace_linp/ai_researcher/researcher/integrations/opencode-sdk-python-next/src",
    )
    from opencode_ai import AsyncOpencode

    OPENCODE_AVAILABLE = True
except ImportError as e:
    OPENCODE_AVAILABLE = False
    logging.warning(f"opencode-ai next branch package not available: {e}")


class OpenCodeClient:
    """
    Wrapper for OpenCode SDK with session management.

    Provides high-level interface for:
    - Creating and managing sessions
    - Sending instructions and receiving responses
    - Parsing tool execution results
    - Handling errors and retries
    """

    def __init__(
        self,
        base_url: str = "http://localhost:4096",
        timeout: float = 300.0,
        config_file: Optional[Path] = None
    ):

        if not OPENCODE_AVAILABLE:
            raise ImportError("opencode-ai next branch package is required.")

        self.client = AsyncOpencode(base_url=base_url, timeout=timeout)
        self.session_id: Optional[str] = None
        self.logger = logging.getLogger(__name__)

        self.config = self._load_config(config_file)
        self.default_model = self.config.get("model", "/home/ai_researcher/.cache/modelscope/hub/models/Qwen/Qwen3-Coder-30B-A3B-Instruct")
        self.default_provider = self._extract_provider_from_model(self.default_model)

    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        if config_file and config_file.exists():
            try:
                import json
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load config file {config_file}: {e}")

        # Default configuration
        return {
            "model": "qwen-local//home/ai_researcher/.cache/modelscope/hub/models/Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "provider": {
                "qwen-local": {
                    "models": {
                        "/home/ai_researcher/.cache/modelscope/hub/models/Qwen/Qwen3-Coder-30B-A3B-Instruct": {
                            "options": {
                                "baseUrl": "http://localhost:8001/v1",
                                "apiKey": "EMPTY"
                            }
                        }
                    }
                }
            }
        }

    def _extract_provider_from_model(self, model_id: str) -> str:
        """
        Extract provider ID from model ID.

        Args:
            model_id: Model identifier (can be provider/model or full path)

        Returns:
            Provider ID
        """
        if model_id.startswith("/"):
            return "qwen-local"
        # Otherwise, extract from provider/model format
        if "/" in model_id:
            return model_id.split("/")[0]
        return "qwen-local"

    async def create_session(self, work_dir: Path) -> str:
        """
        Create a new OpenCode session.

        Args:
            work_dir: Working directory for the session

        Returns:
            Session ID
        """
        session = await self.client.session.create(
            directory=str(work_dir),
            title=f"Session for {work_dir.name}",
        )
        self.session_id = session.id

        self.logger.info(f"Created OpenCode session: {self.session_id}")
        return self.session_id

    async def send_instruction(
        self,
        instruction: str,
        *,
        files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send instruction to OpenCode and get response.

        OpenCode will:
        1. Understand the instruction
        2. Use tools (read, write, edit, bash) as needed
        3. Execute code and debug automatically
        4. Return results

        Args:
            instruction: Detailed instruction for the task
            files: Optional list of file paths to include in context

        Returns:
            Dict with keys:
                - success: bool
                - output: str (text response)
                - files: List[str] (created/modified files)
                - exit_code: int
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session first.")

        # Try different parts format - maybe SDK expects different structure
        # Format 1: Simple dict (current)
        parts = [{"type": "text", "text": instruction}]

        # Add file references if provided
        if files:
            file_refs = "\n".join([f"@{f}" for f in files])
            parts.append({
                "type": "text",
                "text": f"\n\nRelevant files:\n{file_refs}"
            })

        self.logger.info(f"Sending instruction to OpenCode (session: {self.session_id})")
        print("Sending instruction to OpenCode:", instruction)

        # Parse provider and model from default_model
        if '//' in self.default_model:
            provider_id, model_path = self.default_model.split('/', 1)
            model_id = model_path
        else:
            provider_id, model_id = self.default_model.split('/', 1) if '/' in self.default_model else (self.default_provider, self.default_model)

        self.logger.info(f"Sending to session {self.session_id} with model {model_id}")
        print(f"self.session_id: {self.session_id}")
        print(f"model_id: {model_id}")
        print(f"provider_id: {provider_id}")

        response = await self.client.session.prompt(
            id=self.session_id,
            parts=parts,
            model={
                "model_id": model_id,
                "provider_id": provider_id
            },
            tools={
                'bash': True,
                'edit': True,
                'write': True,
                'read': True,
                'grep': True,
                'glob': True,
                'list': True,
                'patch': True
            }
        )

        result = self._parse_response(response)
        self.logger.info(
            f"OpenCode response: success={result['success']}, "
            f"files={len(result['files'])}"
        )

        return result

    def _parse_response(self, response) -> Dict[str, Any]:
        result = {
            "success": True,
            "output": "",
            "files": [],
            "exit_code": 0,
            "tool_calls": []
        }

        # 检查是否有错误
        if hasattr(response, 'info') and response.info.error:
            result["success"] = False
            result["exit_code"] = -1
            result["output"] = f"API Error: {response.info.error}"
            return result

        # 解析parts
        if hasattr(response, 'parts'):
            for part in response.parts:
                if hasattr(part, 'type'):
                    if part.type == "text":
                        # TextPart: see opencode_ai.types.text_part.TextPart
                        result["output"] += getattr(part, 'text', '')

                    elif part.type == "tool":
                        # ToolPart: see opencode_ai.types.tool_part.ToolPart
                        state = getattr(part, "state", None)
                        status = getattr(state, "status", "unknown") if state is not None else "unknown"
                        tool_name = getattr(part, "tool", "")

                        tool_info = {
                            "name": tool_name,
                            "status": status,
                        }

                        if status == "completed":
                            # ToolStateCompleted: output 字段保存工具输出
                            output = getattr(state, "output", "")
                            tool_info["result"] = output
                            result["output"] += f"\n[Tool: {tool_name}] Completed"

                        elif status == "error":
                            # ToolStateError: error 字段保存错误信息
                            result["success"] = False
                            result["exit_code"] = -1
                            error_msg = getattr(state, "error", "Unknown error")
                            result["output"] += f"\n[Tool Error: {tool_name}] {error_msg}"
                            tool_info["error"] = error_msg

                        result["tool_calls"].append(tool_info)

                    elif part.type == "patch":
                        # PatchPart: see opencode_ai.types.part.PatchPart
                        files = getattr(part, "files", None)
                        if isinstance(files, list):
                            result["files"].extend(str(f) for f in files)

        return result

    async def get_messages(self) -> List[Dict[str, Any]]:
        if not self.session_id:
            return []

        messages = await self.client.session.messages(self.session_id)
        return [item.to_dict() for item in messages]

    async def close_session(self):
        """Close current session and cleanup."""
        if self.session_id:
            self.logger.info(f"Closing OpenCode session: {self.session_id}")
            await self.client.session.delete(self.session_id)
            self.session_id = None

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close_session()


def check_opencode_availability() -> bool:
    return OPENCODE_AVAILABLE


async def test_client():
    """Test function for OpenCodeClient with result verification and file logging."""
    client = OpenCodeClient()
    work_dir = Path("/home/ai_researcher/projects/workplace_linp/ai_researcher/tests/test_opencode")
    log_file = work_dir / "opencode_send_result.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 先尝试创建 session，再发送指令
        await client.create_session(work_dir)
        result = await client.send_instruction(
            "Write a simple Python script that prints 'Hello, OpenCode!'"
        )
        success = bool(result.get("success", False))
        output = result.get("output", "")
        exit_code = result.get("exit_code", 0)
    except Exception as e:
        # 包括创建 session 或发送指令出错，都视为失败
        success = False
        output = f"Exception in test_client: {e}"
        exit_code = -1

    # 追加写入验证结果到文件
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("=== OpenCode send_instruction test ===\n")
        f.write(f"success: {success}\n")
        f.write(f"exit_code: {exit_code}\n")
        f.write("output:\n")
        f.write(str(output) + "\n")
        f.write("----------------------------------------\n")

    print(f"发送结果已写入文件: {log_file}")

    # 如需继续查看消息历史，可以解开下面两行注释
    # messages = await client.get_messages()
    # print("Client messages:", messages)

    await client.close_session()
    return {
        "success": success,
        "exit_code": exit_code,
        "output": output,
        "log_file": str(log_file),
    }


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_client())