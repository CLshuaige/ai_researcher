import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


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
