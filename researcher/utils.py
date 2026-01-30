from collections import Counter
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime
import json
import re

#from researcher.config import config
from researcher.exceptions import WorkflowError


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


def get_default_config_path(config_filename: str = "debug.yaml") -> Path:
    """Get default path for configuration files"""
    return get_project_root() / "configs" / config_filename


def clean_markdown_identifiers(content: str) -> str:
    """Remove control identifiers from markdown content that should not be saved"""
    import re
    
    identifiers = [
        r'==========CLEAR==========\s*',
        r'==========UNCLEAR==========\s*',
        r'==========READY==========\s*',
        r'==========NEEDS_REVISION==========\s*',
        r'==STEP_COMPLETE==\s*',
        r'==========STEP_COMPLETE==========\s*',
    ]
    
    cleaned = content
    for pattern in identifiers:
        # Remove identifier and any trailing whitespace/newlines
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
    
    return cleaned


def save_markdown(content: str, filepath: Path) -> None:
    """Save content to markdown file"""
    cleaned_content = clean_markdown_identifiers(content)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)


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


def get_llm_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get standard LLM configuration for autogen
    Load LLM configuration from json file"""
    import json

    if config_path is None:
        config_path = get_default_config_path("llm_config.json")
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            # print(f"Loaded LLM config from {config_path}")
            # print(f"LLM Config: {config_data}")
            return config_data
    else:
        raise WorkflowError(f"LLM config file not found: {config_path}")

def load_global_config(config_path: Optional[Path] = None, config_filename: str = "debug.yaml") -> Dict[str, Any]:
    """Load global configuration from yaml file"""
    import yaml
    
    if config_path is None:
        config_path = get_default_config_path(config_filename)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            # print(f"Loaded global config from {config_path}")
            # print(f"Global Config: {config_data}")
            return config_data
    except Exception as e:
        raise WorkflowError(f"Failed to load global config from {config_path}: {str(e)}")



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
    
def serialize_groupchat_messages(agent_chat_messages) -> dict:
    result = {}

    for manager, messages in agent_chat_messages.items():
        manager_name = getattr(manager, "name", "GroupChatManager")

        # 关键修复点：messages 不是 list，而是 dict
        flattened_messages = []

        if isinstance(messages, dict):
            for _, msg_list in messages.items():
                flattened_messages.extend(msg_list)
        else:
            flattened_messages = messages

        result[manager_name] = [
            {
                "role": m.get("role"),
                "name": m.get("name"),
                "content": m.get("content"),
            }
            for m in flattened_messages
        ]

    return result



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
        history_data["agent_chat_messages"] = serialize_groupchat_messages(agent_chat_messages)

    save_json(history_data, filepath)


def save_session_metadata(workspace_dir: Path, session_data: Dict[str, Any]) -> None:
    """Save session metadata to workspace"""
    session_file = workspace_dir / "session.json"
    save_json(session_data, session_file)


def load_session_metadata(workspace_dir: Path) -> Optional[Dict[str, Any]]:
    """Load session metadata from workspace"""
    session_file = workspace_dir / "session.json"
    return load_json(session_file)


def deduplicate_long_repeats(sender, message, recipient, silent):
    """Hook function for process_message_before_send to deduplicate long repeats in messages.

    Detects and removes duplicate content blocks in individual messages before sending.
    Uses line normalization (digits→placeholder, collapse whitespace) and pattern matching
    to identify exact duplicates and repetitive pattern blocks within a single message.

    For LSH/near-duplicate detection at scale, consider datasketch.MinHashLSH.

    Args:
        sender: The sending ConversableAgent
        message: The message being sent (str or dict)
        recipient: The recipient Agent
        silent: Whether the message is silent

    Returns:
        The processed message (str or dict) with duplicate blocks replaced by summary placeholders
    """
    from autogen import ConversableAgent, Agent
    from typing import Union, Dict, Any

    # Extract content from message
    if isinstance(message, dict):
        content = message.get("content", "")
        if not isinstance(content, str):
            return message
    else:
        content = str(message)

    if not content:
        return message

    def _norm(line: str) -> str:
        """Normalize line: collapse whitespace and replace digits with placeholder"""
        s = re.sub(r"\s+", " ", line.strip())
        if not s:
            return ""

        normalized = re.sub(r"\d+", "0", s)

        if "Loading" in normalized and ("|" in normalized or "%" in normalized):
            # "Loading weights: 99%|█████████▊| 393/398 [00:00<00:00, 6747.06it/s, Materializing param=...]"
            # -> "Loading weights: 0%|██████████| 0/0 [0:0<0:0, 0.0it/s, Materializing param=]"
            normalized = re.sub(r'\d+%\|.*\| \d+/\d+ \[.*?\]', '0%|██████████| 0/0 [0:0<0:0, 0.0it/s', normalized)
            normalized = re.sub(r'Materializing param=.*', 'Materializing param=', normalized)

        # File paths pattern
        elif "/" in normalized or "\\" in normalized:
            # For paths, keep only the general structure
            parts = re.split(r"[\/\\]", normalized)
            for i, part in enumerate(parts):
                if "." in part and any(ext in part for ext in [".jpg", ".png", ".json", ".py", ".txt", ".md", ".wav"]):
                    parts[i] = "file.0"
                elif part.replace("0", "").replace(".", "").replace("-", "").replace("_", "") == "":
                    parts[i] = "0"
                elif "_" in part and any(x in part for x in ["coco", "hateful_memes", "mimic_cxr"]):
                    dataset_match = re.search(r'(coco|hateful_memes|mimic_cxr)_(\d+)', part)
                    if dataset_match:
                        parts[i] = f"{dataset_match.group(1)}_0"
            normalized = "/".join(parts)

        return normalized

    lines = content.split("\n")

    # First pass: normalize all lines and identify frequent patterns
    normalized_lines = [_norm(line) for line in lines]
    pattern_counts = Counter(normalized_lines)

    # Find patterns that appear frequently
    frequent_patterns = {pattern: count for pattern, count in pattern_counts.items() if count > 5}

    if frequent_patterns:
        result: list[str] = []
        i = 0

        while i < len(lines):
            if not lines[i].strip():
                result.append(lines[i])
                i += 1
                continue

            current_norm = normalized_lines[i]

            # Check if this line matches a frequent pattern
            if current_norm in frequent_patterns:
                start = i
                pattern = current_norm
                i += 1
                while i < len(lines) and normalized_lines[i] == pattern:
                    i += 1

                block_size = i - start
                if block_size >= 5:  # Only compress if we have 5+ consecutive similar lines
                    sample_lines = [lines[j].strip() for j in range(start, min(start + 3, i))]
                    sample_text = " | ".join(sample_lines[:2])  # Show first 2 samples
                    if len(sample_lines) > 2:
                        sample_text += " | ..."

                    result.append(f"[Compressed {block_size} similar file paths: {sample_text}]")
                    continue

            result.append(lines[i])
            i += 1
    else:
        # No frequent patterns, keep original content
        result = lines

    new_text = "\n".join(result)

    # Return the modified message
    if isinstance(message, dict):
        if new_text != content:
            modified_message = dict(message)
            modified_message["content"] = new_text
            return modified_message
        else:
            return message
    else:
        return new_text

