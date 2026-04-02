from collections import Counter
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime
import json
import os
import re
import tempfile

import sys
import uuid

from autogen.events.agent_events import GroupChatRunChatEvent, TextEvent, InputRequestEvent, TerminationEvent, RunCompletionEvent, ErrorEvent
from autogen.agentchat import run_group_chat_iter

from researcher.state import ResearchState
from researcher.schemas import ChatResult

#from researcher.config import config
from researcher.exceptions import WorkflowError
from researcher.latex.utils import compile_latex


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


def get_relative_path(target_path: Path, base_dir: Path) -> str:
    """Return POSIX-style relative path"""
    return Path(os.path.relpath(target_path.resolve(), start=base_dir.resolve())).as_posix()


def load_artifact_from_file(workspace_dir: Path, artifact_type: str) -> Optional[str]:
    """Load artifact content from file"""
    artifact_path = get_artifact_path(workspace_dir, artifact_type)
    return load_markdown(artifact_path)


_LLM_CONFIG: Optional[Dict[str, Any]] = None
def set_llm_config_override(config: Optional[Dict[str, Any]]) -> None:
    global _LLM_CONFIG
    _LLM_CONFIG = config


def get_llm_config(config_path: Optional[Path] = None, use_tool: bool = False, sampling_params: dict = None) -> Dict[str, Any]:
    """Get standard LLM configuration for autogen
    Load LLM configuration from json file"""
    import json

    if _LLM_CONFIG is not None:
        config_data = json.loads(json.dumps(_LLM_CONFIG))
    else:
        if config_path is None:
            config_path = get_default_config_path("llm_config.json")

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            raise WorkflowError(f"LLM config file not found: {config_path}")

    if use_tool:
        for config in config_data["config_list"]:
            config["tool_choice"] = "required"

    if sampling_params:
        for config in config_data["config_list"]:
            for k, v in sampling_params.items():
                config[k] = v

    return config_data

def load_global_config(config_path: Optional[Path] = None, config_filename: str = "debug.yaml") -> Dict[str, Any]:
    """Load global configuration from yaml file"""
    import yaml
    
    if config_path is None:
        config_path = get_default_config_path(config_filename)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            return config_data
    except Exception as e:
        raise WorkflowError(f"Failed to load global config from {config_path}: {str(e)}")



def merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:

    merged = dict(base)
    for key, value in patch.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def update_llm_config(llm_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:

    updated_config = llm_config.copy()
    for key, value in new_config.items():
        if value is not None:
            updated_config[key] = value
    return updated_config


def parse_json_from_response(response: str) -> Dict[str, Any]:

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
        if isinstance(manager, str) and manager.strip():
            manager_name = manager
        else:
            manager_name = getattr(manager, "name", None) or "GroupChatManager"

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

def iterable_group_chat(
    state: ResearchState,
    max_rounds: int = 10,
    enable_hitl: bool = False,
    pattern = None,
    prompt: str = None,   
):
    _publish_event_progress(
        state,
        "iterator_started",
        max_rounds=max_rounds,
        human_in_the_loop=enable_hitl,
    )
    iterator = run_group_chat_iter(
        pattern=pattern,
        messages=prompt,
        max_rounds=max_rounds,
        yield_on=[GroupChatRunChatEvent, TextEvent, InputRequestEvent, TerminationEvent, RunCompletionEvent, ErrorEvent]
    )

    global_history = []
    for step, event in enumerate(iterator, start=1):
        if isinstance(event, GroupChatRunChatEvent):
            speaker = str(event.content.speaker)
            print(f"\n{'─' * 60}")
            print(f"🎭 [{step}] {speaker}'s turn")
            print('─' * 60)
            _publish_event_progress(
                state,
                "turn_started",
                step=step,
                speaker=speaker,
            )
        elif isinstance(event, TextEvent):
            sender = str(event.content.sender)
            content = str(event.content.content)
            print(f"💬 {sender}: {content}")
            global_history.append({
                "name": sender,
                "content": content
            })
            _publish_event_progress(
                state,
                "message",
                step=step,
                sender=sender,
                content_preview=content,
            )
        elif isinstance(event, InputRequestEvent):
            prompt_text = str(event.content.prompt)
            request_id = str(uuid.uuid4())
            _publish_event_progress(
                state,
                "input_requested",
                step=step,
                request_id=request_id,
                prompt=prompt_text,
            )
            # api call
            # 1. wait for client input
            user_input = _wait_for_user_input(request_id)
            # 2. input from cli
            # user_input = input(prompt_text)
            event.content.respond(user_input)
            _publish_event_progress(
                state,
                "input_submitted",
                step=step,
                input_length=len(user_input),
                #user_input=user_input
            )
        elif isinstance(event, TerminationEvent):
            _publish_event_progress(
                state,
                "termination",
                step=step,
                detail=event.content.termination_reason,
            )
        elif isinstance(event, RunCompletionEvent):
            result_history = event.content.history
            summary = event.content.summary
            context_vars = event.content.context_variables
            last_speaker = event.content.last_speaker
            _publish_event_progress(
                state,
                "run_completion",
                step=step,
                history_length=len(result_history),
                last_speaker=str(last_speaker) if last_speaker else None,
                summary_preview=summary,
            )

    result = ChatResult(chat_history=global_history)
    context = pattern.context_variables

    # 打印总结
    print(f"\n{'=' * 60}")
    print(f"✅ Chat completed - {len(global_history)} messages")
    print(f"📊 Context variables: {list(context.keys()) if context else 'None'}")
    print('=' * 60)

    return result, context, None

def _publish_event_progress(
    state: ResearchState,
    progress_event: str,
    **extra: Any,
) -> None:
    """Publish per-iteration progress when API runtime is active."""
    project_id = state.get("project_id")
    node = state.get("start_node")
    stage = state.get("stage")
    if not project_id:
        return

    app_module = sys.modules.get("researcher.api.app")
    if app_module is None:
        return

    fastapi_app = getattr(app_module, "app", None)
    event_bus = None
    if fastapi_app is not None:
        event_bus = getattr(getattr(fastapi_app, "state", None), "event_bus", None)
    if event_bus is None:
        event_bus = getattr(app_module, "event_bus", None)
    if event_bus is None:
        return

    payload = {
        "event": "node_progress",
        "project_id": project_id,
        "node": node,
        "stage": stage,
        "progress_event": progress_event,
        "timestamp": datetime.now().isoformat(),
    }
    payload.update(extra)

    try:
        event_bus.publish(project_id, payload)
    except Exception:
        # Progress publishing must not break workflow execution.
        pass

def _wait_for_user_input(request_id: str):

    app_module = sys.modules.get("researcher.api.app")

    if not app_module:
        raise RuntimeError("API runtime not available")

    input_store = getattr(app_module, "input_store")

    input_store.create(request_id)

    user_input = input_store.wait_for_input(request_id)

    return user_input


# markdown -> html ->PDF
def _render_markdown_math(markdown_text: str) -> str:
    import latex2mathml.converter

    def render_block(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        if not expression:
            return match.group(0)
        return f'\n<div class="math-block">{latex2mathml.converter.convert(expression)}</div>\n'

    def render_inline(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        if not expression:
            return match.group(0)
        return f'<span class="math-inline">{latex2mathml.converter.convert(expression)}</span>'

    segments = re.split(r"(```[\s\S]*?```|`[^`\n]+`)", markdown_text)
    rendered_segments = []

    for segment in segments:
        if segment.startswith("```") or (segment.startswith("`") and segment.endswith("`")):
            rendered_segments.append(segment)
            continue

        segment = re.sub(r"\$\$([\s\S]+?)\$\$", render_block, segment)
        segment = re.sub(r"(?<!\\)\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)", render_inline, segment)
        rendered_segments.append(segment)

    return "".join(rendered_segments)


def _rewrite_html_local_image_sources(html: str, base_dir: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for image in soup.find_all("img"):
        src = str(image.get("src", "")).strip()
        if not src or re.match(r"^(?:https?:|data:|file:)", src):
            continue
        resolved = (base_dir / src).resolve()
        if resolved.exists():
            image["src"] = resolved.as_uri()
    return str(soup)


def build_markdown_pdf_html(markdown_text: str, base_dir: Path, title: str) -> str:
    import markdown as md

    markdown_text = _render_markdown_math(markdown_text)
    body = md.markdown(
        markdown_text,
        extensions=["extra", "fenced_code", "tables", "sane_lists"],
    )
    body = _rewrite_html_local_image_sources(body, base_dir)
    mermaid_uri = (get_project_root() / "researcher" / "static" / "mermaid.min.js").resolve().as_uri()
    base_uri = base_dir.resolve().as_uri()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <base href="{base_uri}/">
  <style>
    body {{
      font-family: "Noto Serif CJK SC", "Noto Serif", "Times New Roman", serif;
      color: #111827;
      line-height: 1.75;
      margin: 0 auto;
      max-width: 860px;
      padding: 32px 40px 56px;
      font-size: 14px;
      background: #ffffff;
    }}
    h1, h2, h3, h4 {{
      color: #0f172a;
      line-height: 1.3;
      margin-top: 1.5em;
      margin-bottom: 0.6em;
    }}
    p, ul, ol, table, pre, blockquote {{
      margin-top: 0.8em;
      margin-bottom: 0.8em;
    }}
    img {{
      display: block;
      max-width: min(88%, 720px);
      max-height: 460px;
      width: auto;
      height: auto;
      margin: 16px auto;
      object-fit: contain;
      page-break-inside: avoid;
    }}
    .math-inline {{
      display: inline-block;
      vertical-align: middle;
      max-width: 100%;
    }}
    .math-inline math {{
      font-size: 1em;
    }}
    .math-block {{
      display: block;
      margin: 16px 0;
      text-align: center;
      page-break-inside: avoid;
    }}
    .math-block math {{
      display: inline-block;
      max-width: 100%;
      font-size: 1.05em;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #cbd5e1;
      padding: 8px 10px;
      vertical-align: top;
    }}
    th {{
      background: #f8fafc;
    }}
    pre {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      overflow-x: auto;
      padding: 12px 14px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    code {{
      font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
      font-size: 0.92em;
    }}
    blockquote {{
      border-left: 4px solid #94a3b8;
      margin-left: 0;
      padding-left: 16px;
      color: #334155;
    }}
    .mermaid {{
      margin: 20px 0;
      text-align: center;
      page-break-inside: avoid;
    }}
    .mermaid svg {{
      max-width: 100%;
      height: auto;
    }}
  </style>
  <script src="{mermaid_uri}"></script>
  <script>
    window.__MERMAID_READY = false;
    window.__MERMAID_ERROR = null;
    window.addEventListener("load", async () => {{
      try {{
        document.querySelectorAll("pre > code.language-mermaid").forEach((code) => {{
          const container = document.createElement("div");
          container.className = "mermaid";
          container.textContent = code.textContent;
          code.parentElement.replaceWith(container);
        }});
        if (document.querySelector(".mermaid")) {{
          if (typeof mermaid === "undefined") {{
            throw new Error("Mermaid library failed to load");
          }}
          mermaid.initialize({{ startOnLoad: false, securityLevel: "loose", theme: "default" }});
          await mermaid.run({{ querySelector: ".mermaid" }});
        }}
      }} catch (error) {{
        window.__MERMAID_ERROR = String(error);
      }} finally {{
        window.__MERMAID_READY = true;
      }}
    }});
  </script>
</head>
<body>
{body}
</body>
</html>
"""


def markdown_to_pdf(markdown_path: Path, pdf_path: Optional[Path] = None, title: Optional[str] = None) -> Path:
    from playwright.sync_api import sync_playwright

    markdown_path = Path(markdown_path)
    pdf_path = Path(pdf_path) if pdf_path else markdown_path.with_suffix(".pdf")
    markdown_text = load_markdown(markdown_path)

    html = build_markdown_pdf_html(
        markdown_text=markdown_text,
        base_dir=markdown_path.parent,
        title=title or markdown_path.stem,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "render.html"
        html_path.write_text(html, encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.wait_for_function("window.__MERMAID_READY === true", timeout=10000)
            mermaid_error = page.evaluate("window.__MERMAID_ERROR")
            if mermaid_error:
                browser.close()
                raise WorkflowError(f"Mermaid render failed for {markdown_path.name}: {mermaid_error}")
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={
                    "top": "18mm",
                    "right": "16mm",
                    "bottom": "18mm",
                    "left": "16mm",
                },
            )
            browser.close()

    return pdf_path


def latex_to_pdf(latex_path: Path):
    with open(latex_path, "r", encoding="utf-8") as f:
        latex_content = f.read()
    
    # clear the auxiliary files generated by previous compilations to avoid interference
    aux_extensions = [".aux", ".log", ".out", ".toc", ".synctex.gz"]
    for ext in aux_extensions:
        aux_file = latex_path.with_suffix(ext)
        if aux_file.exists():
            aux_file.unlink()
    sueccess, pdf_path = compile_latex(latex_path, latex_content)
    if sueccess:
        print("Compile succeed!")