from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime
import json

#from researcher.config import config
from researcher.exceptions import WorkflowError


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


def get_llm_config() -> Dict[str, Any]:
    """Get standard LLM configuration for autogen
    Load LLM configuration from json file"""
    import json

    config_path = Path(__file__).parent.parent / "configs" / "llm_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            # print(f"Loaded LLM config from {config_path}")
            # print(f"LLM Config: {config_data}")
            return config_data

def load_global_config(config_path: Path) -> Dict[str, Any]:
    """Load global configuration from yaml file"""
    import yaml
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            # print(f"Loaded global config from {config_path}")
            # print(f"Global Config: {config_data}")
            return config_data
    except Exception as e:
        raise WorkflowError(f"Failed to load global config: {str(e)}")



def update_llm_config(llm_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
    """Update LLM configuration with new settings"""
    updated_config = llm_config.copy()
    for key, value in new_config.items():
        if value is not None:
            updated_config[key] = value
    return updated_config


def parse_json_from_response(response: str) -> Dict[str, Any]:
    """Extract and parse JSON from LLM response"""
    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise WorkflowError(f"Failed to parse JSON: {str(e)}\nResponse: {response}")


def save_agent_history(
    workspace_dir: Path,
    node_name: str,
    messages: list,
    agent_chat_messages: Optional[Dict] = None
) -> None:
    """Save AG2 agent conversation history

    Args:
        workspace_dir: Project workspace directory
        node_name: Name of the node (e.g., 'hypothesis_construction')
        messages: GroupChat messages or single agent messages
        agent_chat_messages: Optional dict of agent.chat_messages
    """
    history_dir = workspace_dir / "history" / node_name
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = history_dir / f"{timestamp}.json"

    history_data = {
        "timestamp": datetime.now().isoformat(),
        "node": node_name,
        "messages": messages,
    }

    if agent_chat_messages:
        # Convert agent names to strings for JSON serialization
        history_data["agent_chat_messages"] = {
            str(k): v for k, v in agent_chat_messages.items()
        }

    save_json(history_data, filepath)


def save_session_metadata(workspace_dir: Path, session_data: Dict[str, Any]) -> None:
    """Save session metadata to workspace"""
    session_file = workspace_dir / "session.json"
    save_json(session_data, session_file)


def load_session_metadata(workspace_dir: Path) -> Optional[Dict[str, Any]]:
    """Load session metadata from workspace"""
    session_file = workspace_dir / "session.json"
    return load_json(session_file)

