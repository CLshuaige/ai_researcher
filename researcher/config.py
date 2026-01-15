import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class ModelConfig(BaseModel):
    """LLM model configuration"""
    provider: str = Field(default="openai", description="Model provider: openai, anthropic, etc.")
    model_name: str = Field(default="gpt-4", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    base_url: Optional[str] = Field(default=None, description="API base URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)


class DebateConfig(BaseModel):
    """Debate configuration for hypothesis and method design"""
    max_rounds: int = Field(default=3, ge=1, description="Maximum debate rounds")
    agreement_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Agreement threshold to stop debate")


class WorkspaceConfig(BaseModel):
    """Workspace directory configuration"""
    base_dir: Path = Field(default=Path("./workspace"), description="Base workspace directory")

    def get_project_dir(self, project_name: str) -> Path:
        """Generate project-specific workspace directory"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_dir = self.base_dir / f"{timestamp}_{project_name}"
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir


class ResearcherConfig(BaseModel):
    """Main configuration for AI Researcher system"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)

    @classmethod
    def from_registry(cls, model_preset: Optional[str] = None) -> "ResearcherConfig":
        """
        Load configuration using a model preset from the registry.

        Args:
            model_preset: Optional model preset name; if None, use default preset.
        """
        from researcher.llm import get_model_preset, DEFAULT_MODEL_PRESET
        
        preset_name = model_preset or DEFAULT_MODEL_PRESET
        model_config = get_model_preset(preset_name)
        
        return cls(
            model=model_config,
            debate=DebateConfig(
                max_rounds=int(os.getenv("DEBATE_MAX_ROUNDS", "3")),
                agreement_threshold=float(os.getenv("DEBATE_AGREEMENT_THRESHOLD", "0.8"))
            ),
            workspace=WorkspaceConfig(
                base_dir=Path(os.getenv("WORKSPACE_DIR", "./workspace"))
            )
        )


config = ResearcherConfig.from_registry()
