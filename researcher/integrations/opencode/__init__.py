import json
from pathlib import Path
from typing import Dict, Any

from .opencode_executor import OpenCodeExecutor, opencode_codebase_experiment


def get_opencode_config() -> Dict[str, Any]:
    """From configs/opencode.json"""
    config_path = Path(__file__).parent.parent.parent / "configs" / "opencode.json"
    if not config_path.exists():
        raise FileNotFoundError(f"OpenCode config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_opencode_server_url() -> str:
    """
    Get OpenCode server URL from configuration.

    Returns:
        OpenCode server URL (e.g., "http://127.0.0.1:4096")
    """
    config = get_opencode_config()
    server_config = config.get("server", {})
    host = server_config.get("host", "127.0.0.1")
    port = server_config.get("port", 4096)
    return f"http://{host}:{port}"


__all__ = [
    "OpenCodeExecutor",
    "opencode_codebase_experiment",
    "get_opencode_config",
    "get_opencode_server_url",
]