import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field
from dotenv import dotenv_values, load_dotenv, set_key


# Processing for secrets

RUNTIME_SECRET_ENV_FILENAME = ".env.secrets"
_NON_SECRET_SENTINELS = {"", "EMPTY", "DUMMY", "DUMMY-KEY", "DUMMY_KEY", "NONE", "NULL"}


def get_runtime_secret_env_path() -> Path:
    return Path(__file__).resolve().parents[1] / RUNTIME_SECRET_ENV_FILENAME


def _is_secret_value_key(key: Any) -> bool:
    # Matches fields that carry a plaintext secret from the frontend/API payload.
    text = str(key or "")
    lower = text.lower()
    return (
        lower == "api_key"
        or lower.endswith("_api_key")
        or text == "apiKey"
        or text.endswith("ApiKey")
    )


def _is_secret_ref_key(key: Any) -> bool:
    # Matches fields that carry only the env-var reference saved after sanitization.
    text = str(key or "")
    lower = text.lower()
    return (
        lower == "api_key_env"
        or lower.endswith("_api_key_env")
        or text == "apiKeyEnv"
        or text.endswith("ApiKeyEnv")
    )


def _secret_ref_key_for(secret_key: str) -> str:
    if secret_key.endswith("Env") or secret_key.endswith("_env"):
        return secret_key
    if secret_key.endswith("apiKey") or secret_key == "apiKey":
        return f"{secret_key}Env"
    return f"{secret_key}_env"


def _secret_value_key_for_ref(ref_key: str) -> str:
    if ref_key.endswith("Env"):
        return ref_key[:-3]
    if ref_key.endswith("_env"):
        return ref_key[:-4]
    return ref_key


def _is_real_secret_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.upper() not in _NON_SECRET_SENTINELS


def _build_secret_env_name(path_parts: Sequence[Any]) -> str:
    # Builds a stable env-var name from config path segments for .env.secrets storage.
    tokens = []
    for part in path_parts:
        raw = str(part).strip()
        if not raw:
            continue
        tokens.append(raw)
    env_name = "_".join(tokens).upper()
    env_name = re.sub(r"[^A-Z0-9_]+", "_", env_name)
    env_name = re.sub(r"_+", "_", env_name).strip("_")
    return env_name or "AIR_SECRET"


def load_runtime_secret_values() -> Dict[str, str]:
    """Load hidden runtime secrets, letting real environment variables win."""
    file_values: Dict[str, str] = {}
    env_path = get_runtime_secret_env_path()
    if env_path.exists():
        file_values.update(
            {
                str(key): str(value)
                for key, value in dotenv_values(env_path).items()
                if key and value is not None
            }
        )
    merged = dict(file_values)
    merged.update({key: value for key, value in os.environ.items() if value is not None})
    return merged


def _persist_runtime_secret(env_name: str, value: str) -> None:
    # Writes a real secret into .env.secrets and mirrors it into the current process env.
    env_path = get_runtime_secret_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if not env_path.exists():
        env_path.touch(mode=0o600)
    else:
        try:
            env_path.chmod(0o600)
        except OSError:
            pass
    set_key(str(env_path), env_name, value, quote_mode="never")
    os.environ[env_name] = value


def persist_config_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Replace plaintext API keys in config with env refs and persist the real values."""
    data = deepcopy(config)

    def transform(value: Any, path: List[Any]) -> Any:
        if isinstance(value, list):
            return [transform(item, [*path, idx]) for idx, item in enumerate(value)]
        if not isinstance(value, dict):
            return value

        result: Dict[str, Any] = {}
        raw_secret_keys = {
            str(key)
            for key in value.keys()
            if _is_secret_value_key(key)
        }

        for key, raw in value.items():
            if _is_secret_ref_key(key):
                if _secret_value_key_for_ref(str(key)) in raw_secret_keys:
                    continue
                ref_text = str(raw).strip()
                if ref_text:
                    result[str(key)] = ref_text
                continue

            if _is_secret_value_key(key):
                if _is_real_secret_value(raw):
                    env_name = _build_secret_env_name([*path, key])
                    _persist_runtime_secret(env_name, str(raw))
                    result[_secret_ref_key_for(str(key))] = env_name
                else:
                    raw_text = "" if raw is None else str(raw)
                    if raw_text:
                        result[str(key)] = raw_text
                continue

            result[str(key)] = transform(raw, [*path, key])

        return result

    return transform(data, [])


def resolve_config_secret_refs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Hydrate api_key_env-style refs back into runtime api_key fields before use."""
    data = deepcopy(config)
    secret_values = load_runtime_secret_values()

    def transform(value: Any) -> Any:
        if isinstance(value, list):
            return [transform(item) for item in value]
        if not isinstance(value, dict):
            return value

        result: Dict[str, Any] = {}
        pending_secret_values: Dict[str, str] = {}

        for key, raw in value.items():
            if _is_secret_ref_key(key):
                ref_text = str(raw).strip()
                if ref_text:
                    result[str(key)] = ref_text
                    secret_value = secret_values.get(ref_text)
                    if secret_value is not None:
                        pending_secret_values[_secret_value_key_for_ref(str(key))] = secret_value
                continue
            result[str(key)] = transform(raw)

        for raw_key, raw_value in pending_secret_values.items():
            current = result.get(raw_key)
            if not _is_real_secret_value(current):
                result[raw_key] = raw_value

        return result

    return transform(data)


# Processing for environment-backed defaults
load_dotenv()
load_dotenv(get_runtime_secret_env_path(), override=False)


class ModelConfig(BaseModel):
    """LLM model configuration"""
    provider: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# Model configuration
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
MODEL_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE")) if os.getenv("MODEL_TEMPERATURE") else None
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS")) if os.getenv("MODEL_MAX_TOKENS") else None

# Workspace configuration
WORKSPACE_BASE_DIR = Path(os.getenv("WORKSPACE_DIR", "./workspace"))

# Node-specific configuration
DEBATE_MAX_ROUNDS = int(os.getenv("DEBATE_MAX_ROUNDS", "3"))
LITERATURE_MAX_PAPERS = int(os.getenv("LITERATURE_MAX_PAPERS", "15"))
EXPERIMENT_TIMEOUT = int(os.getenv("EXPERIMENT_TIMEOUT", "3600"))


def get_model_config() -> ModelConfig:
    """Get current model configuration"""
    return ModelConfig(
        provider=MODEL_PROVIDER,
        model_name=MODEL_NAME,
        api_key=MODEL_API_KEY,
        base_url=MODEL_BASE_URL,
        temperature=MODEL_TEMPERATURE,
        max_tokens=MODEL_MAX_TOKENS
    )


def get_workspace_dir(project_name: str) -> Path:
    """Generate timestamped workspace directory"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = WORKSPACE_BASE_DIR / f"{timestamp}_{project_name}"
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def update_model_config(model_config: ModelConfig):
    """Update global model configuration"""
    global MODEL_PROVIDER, MODEL_NAME, MODEL_API_KEY, MODEL_BASE_URL
    global MODEL_TEMPERATURE, MODEL_MAX_TOKENS

    MODEL_PROVIDER = model_config.provider
    MODEL_NAME = model_config.model_name
    MODEL_API_KEY = model_config.api_key
    MODEL_BASE_URL = model_config.base_url
    MODEL_TEMPERATURE = model_config.temperature
    MODEL_MAX_TOKENS = model_config.max_tokens
