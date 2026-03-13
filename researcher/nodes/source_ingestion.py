from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from autogen import ConversableAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import ContextVariables, ReplyResult, TerminateTarget
from autogen.agentchat.group.patterns import DefaultPattern

from researcher.exceptions import WorkflowError
from researcher.prompts.templates import (
    SOURCE_DOWNLOADER_ITEM_PROMPT,
    SOURCE_DOWNLOADER_SYSTEM_PROMPT,
    SOURCE_INGESTION_ITEM_PROMPT,
    SOURCE_SUMMARIZER_SYSTEM_PROMPT,
)
from researcher.state import ResearchState
from researcher.utils import (
    get_llm_config,
    load_artifact_from_file,
    log_stage,
    save_agent_history,
    save_json,
    save_markdown,
)


def source_ingestion_node(state: ResearchState) -> Dict[str, Any]:
    workspace_dir = state["workspace_dir"]
    log_stage(workspace_dir, "source_ingestion", "Starting source ingestion")

    try:
        input_text = load_artifact_from_file(workspace_dir, "input")
        if not input_text:
            raise WorkflowError("Input file not found")

        config = state["config"]["researcher"]["source_ingestion"]
        max_items = config["max_items"]
        max_files_per_item = config["max_files_per_item"]
        max_depth = config["max_depth"]
        max_bytes_per_file = config["max_bytes_per_file"]
        max_total_bytes_per_item = config["max_total_bytes_per_item"]
        max_preview_rows = config["max_preview_rows"]
        timeout_seconds = config["timeout_seconds"]
        max_rounds_download = config["max_rounds_download"]
        max_rounds_summary = config["max_rounds_summary"]

        sources = _extract_sources_from_input(input_text)
        if not sources:
            raise WorkflowError("No sources found in input.md section: ## Sources")

        if max_items is not None and len(sources) > max_items:
            sources = sources[:max_items]
            log_stage(
                workspace_dir,
                "source_ingestion",
                f"Source list truncated to max_items={max_items}",
            )

        sources_root = workspace_dir / "inputs" / "sources"
        knowledge_dir = workspace_dir / "knowledge"
        sources_root.mkdir(parents=True, exist_ok=True)
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        llm_config = get_llm_config()

        metadata_items: List[Dict[str, Any]] = []
        summary_blocks: List[str] = []
        history_messages: List[Dict[str, Any]] = []

        for idx, source in enumerate(sources, start=1):
            item_id = _safe_item_id(idx, source)
            item_meta: Dict[str, Any] = {
                "item_id": item_id,
                "source_input": source,
                "status": "failed",
                "source_type": None,
                "local_path": None,
                "errors": [],
            }

            try:
                download_state: Dict[str, Any] = {}
                read_budget = {"used": 0}

                def tool_clone_git_repo(
                    git_url: Annotated[str, "Git repository URL"],
                ) -> ReplyResult:
                    repo_path = sources_root / item_id / "repo"
                    local_path = _clone_git_repo(git_url, repo_path, timeout_seconds=timeout_seconds)
                    download_state["source_type"] = "git_url"
                    download_state["resolved_from"] = git_url
                    download_state["local_path"] = str(local_path)
                    return ReplyResult(message="Git repository cloned")

                def tool_download_url(
                    url: Annotated[str, "HTTP/HTTPS URL to download"],
                ) -> ReplyResult:
                    download_dir = sources_root / item_id / "downloads"
                    local_path = _download_url(
                        url,
                        destination_dir=download_dir,
                        timeout_seconds=timeout_seconds,
                        max_total_bytes=max_total_bytes_per_item,
                    )
                    download_state["source_type"] = "url"
                    download_state["resolved_from"] = url
                    download_state["local_path"] = str(local_path)
                    return ReplyResult(message="URL downloaded")

                def tool_resolve_local_path(
                    source_path: Annotated[str, "Local file or directory path"],
                ) -> ReplyResult:
                    local_path = _resolve_local_path(source_path, workspace_dir)
                    download_state["source_type"] = "local_path"
                    download_state["local_path"] = str(local_path)
                    return ReplyResult(message="Local path resolved")

                # downloader
                downloader = ConversableAgent(
                    name="SourceDownloader",
                    system_message=SOURCE_DOWNLOADER_SYSTEM_PROMPT,
                    llm_config=llm_config,
                    functions=[tool_clone_git_repo, tool_download_url, tool_resolve_local_path],
                )
                downloader_ctx = ContextVariables(data={"source_input": source})
                download_pattern = DefaultPattern(
                    initial_agent=downloader,
                    agents=[downloader],
                    context_variables=downloader_ctx,
                    group_manager_args={"llm_config": llm_config},
                )
                downloader.handoffs.set_after_work(TerminateTarget())

                download_prompt = SOURCE_DOWNLOADER_ITEM_PROMPT.format(source_input=source)
                download_kwargs = {}
                download_kwargs["max_rounds"] = max_rounds_download if max_rounds_download is not None else 3
                download_result, _, _ = initiate_group_chat(
                    pattern=download_pattern,
                    messages=download_prompt,
                    **download_kwargs,
                )

                local_path = download_state.get("local_path")
                if not local_path:
                    raise WorkflowError("SourceDownloader did not resolve local_path")
                root_path = Path(local_path).resolve()
                root_is_file = root_path.is_file()

                history_messages.append(
                    {
                        "name": "system",
                        "content": f"=== source_download_start {item_id} ===",
                    }
                )
                history_messages.extend(download_result.chat_history)
                history_messages.append(
                    {
                        "name": "system",
                        "content": f"=== source_download_end {item_id} ===",
                    }
                )

                item_meta["source_type"] = download_state.get("source_type")
                item_meta["resolved_from"] = download_state.get("resolved_from")
                item_meta["local_path"] = str(local_path)

                snapshot = _collect_path_snapshot(
                    Path(local_path),
                    max_files=max_files_per_item,
                    max_depth=max_depth,
                )
                key_files = _pick_key_files(snapshot.get("files", []), limit=8)
                item_meta["snapshot"] = {
                    "kind": snapshot.get("kind"),
                    "file_count": snapshot.get("file_count", 0),
                    "truncated": snapshot.get("truncated", False),
                    "tree": snapshot.get("tree", []),
                    "key_files": key_files,
                }

                def _resolve_allowed_path(path_str: str) -> Path:
                    raw_path = Path(path_str).expanduser()
                    if raw_path.is_absolute():
                        candidate = raw_path.resolve()
                    else:
                        candidate_base = root_path if root_path.is_dir() else root_path.parent
                        candidate = (candidate_base / raw_path).resolve()

                    if root_is_file:
                        if candidate != root_path:
                            raise ValueError("Path is outside allowed source scope")
                    else:
                        if not candidate.is_relative_to(root_path):
                            raise ValueError("Path is outside allowed source scope")
                    return candidate

                def tool_list_structure(
                    path: Annotated[str, "Path to inspect; default should be local_path"],
                    max_depth_arg: Annotated[int, "Depth limit"] = max_depth,
                    max_files_arg: Annotated[int, "File count limit"] = max_files_per_item,
                ) -> Dict[str, Any]:
                    target = _resolve_allowed_path(path)
                    max_files_limit = max_files_arg
                    if max_files_arg is not None and max_files_per_item is not None:
                        max_files_limit = min(max_files_arg, max_files_per_item)
                    elif max_files_per_item is not None:
                        max_files_limit = max_files_per_item

                    max_depth_limit = max_depth_arg
                    if max_depth_arg is not None and max_depth is not None:
                        max_depth_limit = min(max_depth_arg, max_depth)
                    elif max_depth is not None:
                        max_depth_limit = max_depth

                    return _collect_path_snapshot(
                        target,
                        max_files=max_files_limit,
                        max_depth=max_depth_limit,
                    )

                def tool_read_text(
                    path: Annotated[str, "Path to a text/code file"],
                    max_bytes_arg: Annotated[int, "Byte limit for this read"] = max_bytes_per_file,
                ) -> Dict[str, Any]:
                    target = _resolve_allowed_path(path)
                    if not target.exists() or not target.is_file():
                        raise ValueError(f"File not found: {target}")

                    if max_bytes_arg is not None and max_bytes_per_file is not None:
                        per_read_limit = min(max_bytes_arg, max_bytes_per_file)
                    else:
                        per_read_limit = max_bytes_arg if max_bytes_arg is not None else max_bytes_per_file

                    used = read_budget["used"]
                    if max_total_bytes_per_item is not None:
                        remaining = max_total_bytes_per_item - used
                        if remaining <= 0:
                            raise RuntimeError("Read budget exceeded for this source item")
                        effective_limit = remaining if per_read_limit is None else min(per_read_limit, remaining)
                    else:
                        effective_limit = per_read_limit

                    preview = _read_text_preview(target, max_bytes=effective_limit)
                    if max_total_bytes_per_item is not None:
                        consumed = preview["size"] if effective_limit is None else min(preview["size"], effective_limit)
                        read_budget["used"] = used + consumed
                    return preview

                def tool_preview_structured(
                    path: Annotated[str, "Path to csv/tsv/json/jsonl file"],
                    max_rows_arg: Annotated[int, "Row/object preview count"] = max_preview_rows,
                    max_bytes_arg: Annotated[int, "Byte read limit"] = max_bytes_per_file,
                ) -> Dict[str, Any]:
                    target = _resolve_allowed_path(path)
                    if not target.exists() or not target.is_file():
                        raise ValueError(f"File not found: {target}")

                    if max_bytes_arg is not None and max_bytes_per_file is not None:
                        per_read_limit = min(max_bytes_arg, max_bytes_per_file)
                    else:
                        per_read_limit = max_bytes_arg if max_bytes_arg is not None else max_bytes_per_file

                    used = read_budget["used"]
                    if max_total_bytes_per_item is not None:
                        remaining = max_total_bytes_per_item - used
                        if remaining <= 0:
                            raise RuntimeError("Read budget exceeded for this source item")
                        effective_limit = remaining if per_read_limit is None else min(per_read_limit, remaining)
                    else:
                        effective_limit = per_read_limit

                    max_rows_limit = max_rows_arg
                    if max_rows_arg is not None and max_preview_rows is not None:
                        max_rows_limit = min(max_rows_arg, max_preview_rows)
                    elif max_preview_rows is not None:
                        max_rows_limit = max_preview_rows

                    preview = _preview_structured_file(target, max_rows=max_rows_limit, max_bytes=effective_limit)
                    if max_total_bytes_per_item is not None:
                        raw_size = target.stat().st_size
                        consumed = raw_size if effective_limit is None else min(raw_size, effective_limit)
                        read_budget["used"] = used + consumed
                    return preview

                # summarizer
                summarizer = ConversableAgent(
                    name="SourceSummarizer",
                    system_message=SOURCE_SUMMARIZER_SYSTEM_PROMPT,
                    llm_config=llm_config,
                    functions=[tool_list_structure, tool_read_text, tool_preview_structured],
                )
                summarizer.handoffs.set_after_work(TerminateTarget())

                summarize_ctx = ContextVariables(
                    data={
                        "source_input": source,
                        "local_path": str(local_path),
                    }
                )
                summarize_pattern = DefaultPattern(
                    initial_agent=summarizer,
                    agents=[summarizer],
                    context_variables=summarize_ctx,
                    group_manager_args={"llm_config": llm_config},
                )

                prompt = SOURCE_INGESTION_ITEM_PROMPT.format(
                    source_input=source,
                    source_type=item_meta["source_type"],
                    local_path=str(local_path),
                    file_count=snapshot.get("file_count", 0),
                    key_files=", ".join(key_files) if key_files else "(none)",
                )

                summarize_kwargs = {}
                if max_rounds_summary is not None:
                    summarize_kwargs["max_rounds"] = max_rounds_summary
                result, _, _ = initiate_group_chat(
                    pattern=summarize_pattern,
                    messages=prompt,
                    **summarize_kwargs,
                )

                summary_md: Optional[str] = None
                for msg in reversed(result.chat_history):
                    if msg.get("name") == summarizer.name and msg.get("content"):
                        summary_md = str(msg["content"]).strip()
                        break

                if not summary_md:
                    raise WorkflowError("SourceSummarizer did not generate summary")

                item_history = result.chat_history

                item_meta["status"] = "completed"
                item_meta["summary_path"] = "knowledge/knowledge_summary.md"
                summary_blocks.append(
                    "\n".join(
                        [
                            f"## Source {idx}: `{source}`",
                            f"- item_id: `{item_id}`",
                            f"- source_type: `{item_meta['source_type']}`",
                            f"- local_path: `{item_meta['local_path']}`",
                            f"- file_count: `{snapshot.get('file_count', 0)}`",
                            "",
                            summary_md,
                        ]
                    )
                )

                history_messages.append(
                    {
                        "name": "system",
                        "content": f"=== source_item_start {item_id} ===",
                    }
                )
                history_messages.extend(item_history)
                history_messages.append(
                    {
                        "name": "system",
                        "content": f"=== source_item_end {item_id} ===",
                    }
                )

                log_stage(workspace_dir, "source_ingestion", f"Processed source {idx}/{len(sources)}: {source}")

            except Exception as item_error:
                item_meta["errors"].append(str(item_error))
                log_stage(
                    workspace_dir,
                    "source_ingestion",
                    f"Source failed ({source}): {item_error}",
                )

            metadata_items.append(item_meta)

        success_count = sum(1 for item in metadata_items if item.get("status") == "completed")
        failed_count = len(metadata_items) - success_count

        metadata = {
            "generated_at": datetime.now().isoformat(),
            "node": "source_ingestion",
            "config": config,
            "stats": {
                "total": len(metadata_items),
                "successful": success_count,
                "failed": failed_count,
            },
            "items": metadata_items,
        }

        metadata_path = knowledge_dir / "metadata.json"
        save_json(metadata, metadata_path)

        summary_header = [
            "# Source Ingestion Summary",
            "",
            f"- generated_at: `{datetime.now().isoformat()}`",
            f"- total_sources: `{len(metadata_items)}`",
            f"- successful: `{success_count}`",
            f"- failed: `{failed_count}`",
            "",
        ]

        summary_content = "\n\n".join(summary_header + summary_blocks) if summary_blocks else "\n".join(summary_header)

        summary_path = knowledge_dir / "knowledge_summary.md"
        save_markdown(summary_content, summary_path)

        save_agent_history(
            workspace_dir=workspace_dir,
            node_name="source_ingestion",
            messages=history_messages,
        )

        if success_count == 0:
            raise WorkflowError("All source items failed during source ingestion")

        log_stage(workspace_dir, "source_ingestion", f"Completed. success={success_count}, failed={failed_count}")

        update_state = {
            "stage": "source_ingestion",
        }
        # router
        if state["config"]["researcher"]["workflow"] == "default":
            update_state["next_node"] = "end"
        return update_state

    except Exception as e:
        log_stage(workspace_dir, "source_ingestion", f"Error: {str(e)}")
        raise WorkflowError(f"Source ingestion failed: {str(e)}")


TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".py",
    ".ipynb",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".xml",
    ".html",
    ".css",
    ".csv",
    ".tsv",
}

KEY_FILENAME_PRIORITY = {
    "readme.md",
    "readme.txt",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "package.json",
    "cargo.toml",
    "makefile",
    "dockerfile",
    "config.yaml",
    "config.yml",
    "main.py",
}


def _extract_sources_from_input(input_text: str) -> List[str]:
    lines = input_text.splitlines()
    in_sources = False
    sources: List[str] = []

    for line in lines:
        stripped = line.strip()
        heading = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if heading:
            title = heading.group(1).strip().lower()
            if title == "sources":
                in_sources = True
                continue
            if in_sources:
                break

        if not in_sources:
            continue

        bullet = re.match(r"^[-*+]\s+(.+)$", stripped)
        if bullet:
            source = bullet.group(1).strip()
            if source:
                sources.append(source)

    deduped: List[str] = []
    seen = set()
    for source in sources:
        if source not in seen:
            deduped.append(source)
            seen.add(source)
    return deduped


def _safe_item_id(index: int, source: str) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{index:03d}_{digest}"


def _resolve_local_path(source: str, workspace_dir: Path) -> Path:
    source_path = Path(source).expanduser()

    if source_path.is_absolute() and source_path.exists():
        return source_path.resolve()

    workspace_candidate = (workspace_dir / source).expanduser()
    if workspace_candidate.exists():
        return workspace_candidate.resolve()

    cwd_candidate = (Path.cwd() / source).expanduser()
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    if source_path.exists():
        return source_path.resolve()

    raise FileNotFoundError(f"Local path not found: {source}")


def _clone_git_repo(url: str, destination: Path, timeout_seconds: Optional[int]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone", "--depth", "1", url, str(destination)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"git clone failed: {stderr or 'unknown error'}")
    return destination.resolve()


def _download_url(url: str, destination_dir: Path, timeout_seconds: Optional[int], max_total_bytes: Optional[int]) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(url)
    filename = Path(parsed.path).name or "downloaded_file"
    name_suffix = Path(filename).suffix.lower()

    downloaded = 0
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()
        iterator = response.iter_content(chunk_size=8192)
        first_chunk = next(iterator, b"")

        if name_suffix:
            target_path = destination_dir / filename
        else:
            if "application/pdf" in content_type or first_chunk.startswith(b"%PDF-"):
                ext = ".pdf"
            elif "text/html" in content_type:
                ext = ".html"
            elif "application/json" in content_type:
                ext = ".json"
            elif "text/plain" in content_type:
                ext = ".txt"
            elif "application/zip" in content_type:
                ext = ".zip"
            else:
                ext = ".bin"
            target_path = destination_dir / f"{filename}{ext}"

        with open(target_path, "wb") as fp:
            if first_chunk:
                downloaded += len(first_chunk)
                if max_total_bytes is not None and downloaded > max_total_bytes:
                    target_path.unlink(missing_ok=True)
                    raise RuntimeError(f"download exceeded max_total_bytes={max_total_bytes}")
                fp.write(first_chunk)
            for chunk in iterator:
                if not chunk:
                    continue
                downloaded += len(chunk)
                if max_total_bytes is not None and downloaded > max_total_bytes:
                    target_path.unlink(missing_ok=True)
                    raise RuntimeError(f"download exceeded max_total_bytes={max_total_bytes}")
                fp.write(chunk)

    return target_path.resolve()


def _collect_path_snapshot(path: Path, max_files: Optional[int], max_depth: Optional[int]) -> Dict[str, Any]:
    if path.is_file():
        stat = path.stat()
        return {
            "kind": "file",
            "root": str(path),
            "file_count": 1,
            "truncated": False,
            "files": [
                {
                    "path": path.name,
                    "size": stat.st_size,
                    "suffix": path.suffix.lower(),
                }
            ],
            "tree": [path.name],
        }

    file_records: List[Dict[str, Any]] = []
    tree_lines: List[str] = []
    truncated = False

    for current_root, dirnames, filenames in os.walk(path):
        rel_root = Path(current_root).resolve().relative_to(path.resolve())
        depth = len(rel_root.parts)

        if max_depth is not None and depth > max_depth:
            dirnames[:] = []
            continue

        dirnames.sort()
        filenames.sort()

        display_root = "." if str(rel_root) == "." else str(rel_root)
        tree_lines.append(f"{display_root}/")

        if max_depth is not None and depth >= max_depth:
            dirnames[:] = []

        for filename in filenames:
            absolute_file = Path(current_root) / filename
            try:
                stat = absolute_file.stat()
                size = stat.st_size
            except OSError:
                size = None

            rel_path = absolute_file.resolve().relative_to(path.resolve())
            file_records.append(
                {
                    "path": str(rel_path),
                    "size": size,
                    "suffix": absolute_file.suffix.lower(),
                    "name": absolute_file.name,
                }
            )
            tree_lines.append(f"  - {rel_path}")

            if max_files is not None and len(file_records) >= max_files:
                truncated = True
                break

        if truncated:
            break

    return {
        "kind": "directory",
        "root": str(path),
        "file_count": len(file_records),
        "truncated": truncated,
        "files": file_records,
        "tree": tree_lines,
    }


def _pick_key_files(file_records: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    if not file_records:
        return []

    def score(record: Dict[str, Any]) -> tuple:
        name = (record.get("name") or "").lower()
        suffix = (record.get("suffix") or "").lower()
        is_priority_name = 1 if name in KEY_FILENAME_PRIORITY else 0
        is_text = 1 if suffix in TEXT_SUFFIXES else 0
        short_depth = -str(record.get("path", "")).count("/")
        return (is_priority_name, is_text, short_depth)

    ranked = sorted(file_records, key=score, reverse=True)
    return [r["path"] for r in ranked[:limit]]


def _read_text_preview(file_path: Path, max_bytes: Optional[int]) -> Dict[str, Any]:
    file_size = file_path.stat().st_size
    with open(file_path, "rb") as fp:
        payload = fp.read() if max_bytes is None else fp.read(max_bytes)
    truncated = False if max_bytes is None else file_size > max_bytes

    content = payload.decode("utf-8", errors="replace")

    return {
        "path": str(file_path),
        "size": file_size,
        "truncated": truncated,
        "content": content,
    }


def _preview_structured_file(file_path: Path, max_rows: Optional[int], max_bytes: Optional[int]) -> Dict[str, Any]:
    suffix = file_path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as fp:
            reader = csv.DictReader(fp, delimiter=delimiter)
            rows = []
            for idx, row in enumerate(reader):
                if max_rows is not None and idx >= max_rows:
                    break
                rows.append(row)
            return {
                "path": str(file_path),
                "format": suffix[1:],
                "columns": reader.fieldnames or [],
                "rows": rows,
                "preview_rows": len(rows),
            }

    if suffix in {".json", ".jsonl"}:
        file_size = file_path.stat().st_size
        with open(file_path, "rb") as fp:
            payload = fp.read() if max_bytes is None else fp.read(max_bytes)
        truncated = False if max_bytes is None else file_size > max_bytes

        if suffix == ".jsonl":
            text = payload.decode("utf-8", errors="replace")
            lines = [line for line in text.splitlines() if line.strip()]
            if max_rows:
                lines = lines[:max_rows]
            parsed_rows = []
            for line in lines:
                try:
                    parsed_rows.append(json.loads(line))
                except Exception:
                    parsed_rows.append({"raw": line})
            return {
                "path": str(file_path),
                "format": "jsonl",
                "rows": parsed_rows,
                "preview_rows": len(parsed_rows),
                "truncated": truncated,
            }

        text = payload.decode("utf-8", errors="replace")
        if truncated:
            return {
                "path": str(file_path),
                "format": "json",
                "truncated": True,
                "note": "File too large for full JSON parsing within max_bytes limit",
                "preview_text": text[: min(len(text), 2000)],
            }

        data = json.loads(text)
        if isinstance(data, dict):
            keys = list(data.keys())
            if max_rows:
                keys = keys[: max_rows * 2]
                sample = {k: data[k] for k in keys[:max_rows]}
            else:
                sample = data
            return {
                "path": str(file_path),
                "format": "json",
                "kind": "object",
                "keys": keys,
                "sample": sample,
                "truncated": truncated,
            }

        if isinstance(data, list):
            return {
                "path": str(file_path),
                "format": "json",
                "kind": "array",
                "length": len(data),
                "sample": data[:max_rows] if max_rows else data,
                "truncated": truncated,
            }

        return {
            "path": str(file_path),
            "format": "json",
            "kind": type(data).__name__,
            "value": data,
            "truncated": truncated,
        }

    raise ValueError(f"Unsupported structured file type: {suffix}")
