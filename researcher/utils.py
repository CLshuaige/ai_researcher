from pathlib import Path
from typing import Optional, Any
from datetime import datetime
import json


def save_markdown(content: str, filepath: Path) -> None:
    """Save content to markdown file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def load_markdown(filepath: Path) -> Optional[str]:
    """Load content from markdown file"""
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def save_json(data: Any, filepath: Path) -> None:
    """Save data to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: Path) -> Optional[Any]:
    """Load data from JSON file"""
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def initialize_workspace(workspace_dir: Path) -> None:
    """Initialize workspace directory structure"""
    subdirs = [
        "code",
        "data",
        "figures",
        "tex",
        "logs"
    ]
    for subdir in subdirs:
        (workspace_dir / subdir).mkdir(parents=True, exist_ok=True)


def get_artifact_path(workspace_dir: Path, artifact_name: str) -> Path:
    """Get path for specific artifact"""
    artifact_map = {
        "input": "input.md",
        "task": "task.md",
        "literature": "literature.md",
        "idea": "idea.md",
        "method": "method.md",
        "results": "results.md",
        "paper": "paper.pdf",
        "referee": "referee.md"
    }
    return workspace_dir / artifact_map.get(artifact_name, f"{artifact_name}.md")


def log_stage(workspace_dir: Path, stage: str, message: str) -> None:
    """Log stage execution information"""
    log_file = workspace_dir / "logs" / "execution.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] [{stage}] {message}\n")


def load_artifact_from_file(workspace_dir: Path, artifact_type: str) -> Optional[str]:
    """Load artifact content from file"""
    artifact_path = get_artifact_path(workspace_dir, artifact_type)
    return load_markdown(artifact_path)

def load_config_yaml(config_path: Path) -> dict:
    """Load configuration from YAML file"""
    import yaml
    if not config_path.exists():
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

