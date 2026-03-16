from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .opencode_executor import OpenCodeExecutor, opencode_codebase_experiment


def _default_opencode_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "opencode.json"


def get_opencode_config(config_path: Path | None = None) -> Dict[str, Any]:
    path = config_path or _default_opencode_config_path()

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_opencode_server_url() -> str:
    config = get_opencode_config()
    server_config = config.get("server", {})
    host = server_config.get("hostname", "0.0.0.0")
    port = server_config.get("port", 4096)
    return f"http://{host}:{port}"


def list_opencode_model_selectors(config: Dict[str, Any] | None = None) -> List[str]:
    """
    provider_id/model_id

    Example:
      - qwen-local//home/.../Qwen3-Coder-30B-A3B-Instruct
      - openai-free/gpt-5.3-codex
    """
    cfg = config or get_opencode_config()
    providers = cfg.get("provider") or {}
    selectors: List[str] = []

    for provider_id, provider_cfg in providers.items():
        models = (provider_cfg or {}).get("models") or {}
        for model_id in models.keys():
            selectors.append(f"{provider_id}/{model_id}")

    return sorted(selectors)


def resolve_opencode_model_selector(
    model_selector: str,
    config: Dict[str, Any] | None = None,
) -> Tuple[str, str]:

    selector = model_selector.strip()
    cfg = config or get_opencode_config()
    providers = cfg.get("provider") or {}

    for provider_id, provider_cfg in providers.items():
        prefix = f"{provider_id}/"
        if not selector.startswith(prefix):
            continue

        model_id = selector[len(prefix):]
        available_models = (provider_cfg or {}).get("models") or {}
        if model_id in available_models:
            return provider_id, model_id


__all__ = [
    "OpenCodeExecutor",
    "opencode_codebase_experiment",
    "get_opencode_config",
    "get_opencode_server_url",
    "list_opencode_model_selectors",
    "resolve_opencode_model_selector",
]
