from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4
import base64
import os
import re
import tempfile
import zipfile

from pydantic import BaseModel

from researcher.researcher import AIResearcher
from researcher.exceptions import RunCancelledError
from researcher.utils import (
    get_project_root,
    load_global_config,
    save_session_metadata,
    load_session_metadata,
    save_json,
    load_json,
    merge_dict,
)
from researcher.api.events import ProjectEventBus
from researcher.api.schemas import (
    ArtifactContentResponse,
    ArtifactInfo,
    ArtifactUpdateRequest,
    ArtifactUpdateResponse,
    ArtifactsResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    FileContentResponse,
    FileInfo,
    FilesResponse,
    FileUpsertRequest,
    FileWriteResponse,
    LogsResponse,
    NodeHistoryResponse,
    NodeProcess,
    NodeResult,
    ProjectListResponse,
    ProjectCreateRequest,
    ProjectCreateResponse,
    ProjectStatusResponse,
    RunRequest,
    RunResponse,
    RunCancelResponse,
)


NODE_ORDER = [
    "source_ingestion",
    "task_parsing",
    "literature_review",
    "hypothesis_construction",
    "method_design",
    "experiment_execution",
    "report_generation",
    "review",
]


class APIProjectService:
    def __init__(self, base_dir: Optional[Path] = None):
        root = base_dir or (get_project_root() / "workspace" / "api_projects")
        self.base_dir = Path(root)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self._project_locks: Dict[str, Lock] = {}
        self._locks_guard = Lock()
        self._active_runs: Dict[str, Dict[str, Any]] = {}
        self._active_runs_guard = Lock()

    def _get_project_lock(self, project_id: str) -> Lock:
        with self._locks_guard:
            if project_id not in self._project_locks:
                self._project_locks[project_id] = Lock()
            return self._project_locks[project_id]

    def _register_active_run(self, run_id: str, project_id: str) -> Event:
        cancel_event = Event()
        with self._active_runs_guard:
            self._active_runs[run_id] = {
                "project_id": project_id,
                "cancel_event": cancel_event,
            }
        return cancel_event

    def _clear_active_run(self, run_id: str) -> None:
        with self._active_runs_guard:
            self._active_runs.pop(run_id, None)

    def is_run_cancel_requested(self, run_id: str) -> bool:
        with self._active_runs_guard:
            record = self._active_runs.get(run_id)
            if not record:
                return False
            return bool(record["cancel_event"].is_set())

    def cancel_run(self, run_id: str, event_bus: ProjectEventBus) -> RunCancelResponse:
        with self._active_runs_guard:
            record = self._active_runs.get(run_id)
            if not record:
                raise FileNotFoundError(f"Unknown active run_id: {run_id}")
            project_id = str(record["project_id"])
            cancel_event: Event = record["cancel_event"]
            cancel_event.set()

        event_bus.publish(project_id, {
            "event": "run_cancel_requested",
            "project_id": project_id,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
        })

        return RunCancelResponse(
            project_id=project_id,
            run_id=run_id,
            status="cancelling",
        )

    def _safe_slug(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
        return slug or "project"

    def _load_index(self) -> Dict[str, str]:
        return load_json(self.index_path) or {}

    def _save_index(self, index: Dict[str, str]) -> None:
        save_json(index, self.index_path)

    def _resolve_workspace(self, project_id: str) -> Path:
        index = self._load_index()
        raw_path = index.get(project_id)
        if raw_path:
            workspace_dir = Path(raw_path)
            if workspace_dir.exists():
                return workspace_dir

        if not self.base_dir.exists():
            raise FileNotFoundError(f"Unknown project_id: {project_id}")

        for workspace_dir in sorted(self.base_dir.iterdir()):
            if not workspace_dir.is_dir():
                continue
            session = load_session_metadata(workspace_dir) or {}
            discovered_id = str(
                session.get("project_id")
                or session.get("session_id")
                or workspace_dir.name
            )
            if discovered_id != project_id:
                continue
            index[project_id] = str(workspace_dir)
            self._save_index(index)
            return workspace_dir

        raise FileNotFoundError(f"Unknown project_id: {project_id}")

    def _load_project_session(self, workspace_dir: Path) -> Dict[str, Any]:
        session = load_session_metadata(workspace_dir) or {}
        if not session:
            raise FileNotFoundError(f"session.json not found in {workspace_dir}")
        return session

    def _is_project_run_active(self, project_id: str) -> bool:
        lock = self._project_locks.get(project_id)
        return bool(lock and lock.locked())

    def _reconcile_project_session(self, project_id: str, workspace_dir: Path, session: Dict[str, Any]) -> Dict[str, Any]:
        if str(session.get("status") or "").strip().lower() != "running":
            return session
        if self._is_project_run_active(project_id):
            return session

        stage = str(session.get("stage") or "unknown").strip() or "unknown"
        patched = self._patch_project_session(
            workspace_dir,
            {
                "status": "interrupted",
                "stage": stage,
                "sub_stage": None,
                "updated_at": datetime.now().isoformat(),
                "completed_at": session.get("completed_at") or datetime.now().isoformat(),
            },
        )
        return patched

    def _patch_project_session(self, workspace_dir: Path, patch: Dict[str, Any]) -> Dict[str, Any]:
        # Always patch against the latest on-disk session to avoid stale-object overwrite.
        session = load_session_metadata(workspace_dir) or {}
        session.update(patch)
        save_session_metadata(workspace_dir, session)
        return session

    def _serialize(self, obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialize(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._serialize(v) for v in obj]
        return obj

    def _is_text_suffix(self, suffix: str) -> bool:
        return suffix.lower() in {".md", ".txt", ".json", ".log", ".tex", ".py", ".yaml", ".yml", ".csv"}

    def _read_log_tail(self, workspace_dir: Path, tail_lines: int = 25) -> List[str]:
        log_file = workspace_dir / "logs" / "execution.log"
        if not log_file.exists():
            return []
        lines = log_file.read_text(encoding="utf-8").splitlines()
        return lines[-tail_lines:]

    def _resolve_project_file_path(self, project_id: str, file_path: str) -> Path:
        workspace_dir = self._resolve_workspace(project_id).resolve()
        target_path = (workspace_dir / file_path).resolve()
        if not str(target_path).startswith(str(workspace_dir)):
            raise PermissionError("Invalid file path")
        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        return target_path

    def _latest_history_file(self, workspace_dir: Path, node_name: str) -> Optional[Path]:
        history_dir = workspace_dir / "history" / node_name
        if not history_dir.exists():
            return None
        files = sorted(history_dir.glob("*.json"))
        return files[-1] if files else None

    def _count_history_messages(self, history_file: Optional[Path]) -> int:
        if not history_file or not history_file.exists():
            return 0
        data = load_json(history_file) or {}
        messages = data.get("messages") or []
        return len(messages)

    def _artifact_candidates_for_node(self, node_name: str) -> List[str]:
        mapping = {
            "source_ingestion": [
                "knowledge/knowledge.md",
                "knowledge/knowledge_summary.md",
                "knowledge/metadata.json",
            ],
            "task_parsing": ["task.md"],
            "literature_review": ["literature.md"],
            "hypothesis_construction": ["idea.md"],
            "method_design": ["method.md"],
            "experiment_execution": ["results.md"],
            "report_generation": [
                "paper/output.md",
                "paper/main.pdf",
                "paper/main.tex",
                "paper/references.bib",
            ],
            "review": ["referee.md"],
        }
        return mapping.get(node_name, [])

    def _build_node_result(self, workspace_dir: Path, node_name: str, delta: Dict[str, Any]) -> NodeResult:
        session = load_session_metadata(workspace_dir) or {}
        history_file = self._latest_history_file(workspace_dir, node_name)
        artifacts: List[str] = []
        for rel in self._artifact_candidates_for_node(node_name):
            path = workspace_dir / rel
            if path.exists():
                artifacts.append(rel)

        output = self._serialize(delta)
        output["workspace_dir"] = str(workspace_dir)

        stage = str(delta.get("stage") or node_name)
        sub_stage = delta.get("sub_stage")
        if sub_stage is None and str(session.get("stage")) == node_name:
            sub_stage = session.get("sub_stage")
        if "status" in delta:
            status = str(delta.get("status"))
        elif delta.get("error"):
            status = "failed"
        elif delta:
            status = "completed"
        else:
            status = "unknown"

        return NodeResult(
            node=node_name,
            stage=stage,
            sub_stage=str(sub_stage) if sub_stage is not None else None,
            status=status,
            next_node=delta.get("next_node"),
            process=NodeProcess(
                history_path=str(history_file) if history_file else None,
                message_count=self._count_history_messages(history_file),
                log_tail=self._read_log_tail(workspace_dir, tail_lines=20),
            ),
            output=output,
            artifacts=artifacts,
            opencode=self._serialize(delta.get("opencode")),
        )

    def create_project(self, request: ProjectCreateRequest) -> ProjectCreateResponse:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = self._safe_slug(request.project_name)
        project_id = f"{timestamp}_{slug}_{uuid4().hex[:8]}"
        workspace_dir = self.base_dir / project_id
        config_path = Path(request.config_path) if request.config_path else None
        config = load_global_config(config_path=config_path)
        llm_config_path = get_project_root() / "configs" / "llm_config.json"
        llm_config = load_json(llm_config_path) if llm_config_path.exists() else None
        config.setdefault("llm_config", llm_config)
        config.setdefault("researcher", {})
        config["researcher"]["workflow"] = "default" if request.mode == "auto" else "step"

        session_data = {
            "session_id": project_id,
            "project_id": project_id,
            "project_name": request.project_name,
            "workspace_dir": str(workspace_dir),
            "input_text": request.input_text,
            "config": config,
            "model_preset": request.model_preset,
            "run_mode": request.mode,
            "status": "idle",
            "stage": "initialization",
            "sub_stage": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        workspace_dir.mkdir(parents=True, exist_ok=True)
        save_session_metadata(workspace_dir, session_data)

        index = self._load_index()
        index[project_id] = str(workspace_dir)
        self._save_index(index)

        return ProjectCreateResponse(
            project_id=project_id,
            project_name=request.project_name,
            workspace_dir=str(workspace_dir),
            run_mode=request.mode,
            available_nodes=NODE_ORDER,
        )

    def update_project_config(self, project_id: str, request: ConfigUpdateRequest) -> ConfigUpdateResponse:
        workspace_dir = self._resolve_workspace(project_id)
        session = self._load_project_session(workspace_dir)
        current_config = session.get("config") or {}
        merged_config = merge_dict(current_config, request.config_patch)
        self._patch_project_session(
            workspace_dir,
            {
                "config": merged_config,
                "updated_at": datetime.now().isoformat(),
            },
        )

        return ConfigUpdateResponse(
            project_id=project_id,
            applied_patch=request.config_patch,
            researcher_config=merged_config.get("researcher", {}),
        )

    def get_project_status(self, project_id: str) -> ProjectStatusResponse:
        workspace_dir = self._resolve_workspace(project_id)
        session = self._reconcile_project_session(
            project_id,
            workspace_dir,
            self._load_project_session(workspace_dir),
        )
        return ProjectStatusResponse(
            project_id=project_id,
            project_name=session.get("project_name", "unknown"),
            status=session.get("status", "unknown"),
            stage=session.get("stage", "unknown"),
            sub_stage=session.get("sub_stage"),
            run_mode=session.get("run_mode", "step"),
            workspace_dir=str(workspace_dir),
            input_text=session.get("input_text"),
            last_run_id=session.get("last_run_id"),
            updated_at=session.get("updated_at"),
        )

    def list_projects(self) -> ProjectListResponse:
        index = self._load_index()
        new_index = dict(index)
        if self.base_dir.exists():
            for workspace_dir in sorted(self.base_dir.iterdir()):
                if not workspace_dir.is_dir():
                    continue
                session = load_session_metadata(workspace_dir) or {}
                discovered_id = str(
                    session.get("project_id")
                    or session.get("session_id")
                    or workspace_dir.name
                )
                new_index[discovered_id] = str(workspace_dir)
        if new_index != index:
            self._save_index(new_index)

        projects: List[ProjectStatusResponse] = []
        for project_id, raw_path in new_index.items():
            workspace_dir = Path(raw_path)
            if not workspace_dir.exists():
                continue
            session = self._reconcile_project_session(
                project_id,
                workspace_dir,
                load_session_metadata(workspace_dir) or {},
            )
            projects.append(
                ProjectStatusResponse(
                    project_id=str(session.get("project_id") or project_id),
                    project_name=session.get("project_name", "unknown"),
                    status=session.get("status", "unknown"),
                    stage=session.get("stage", "unknown"),
                    sub_stage=session.get("sub_stage"),
                    run_mode=session.get("run_mode", "step"),
                    workspace_dir=str(workspace_dir),
                    input_text=session.get("input_text"),
                    last_run_id=session.get("last_run_id"),
                    updated_at=session.get("updated_at"),
                )
            )

        projects.sort(
            key=lambda item: item.updated_at or "",
            reverse=True,
        )
        return ProjectListResponse(projects=projects, total=len(projects))

    def latest_project(self) -> ProjectStatusResponse:
        project_list = self.list_projects()
        if not project_list.projects:
            raise FileNotFoundError("No projects found")
        return project_list.projects[0]

    def run_project(self, project_id: str, request: RunRequest, event_bus: ProjectEventBus) -> RunResponse:
        lock = self._get_project_lock(project_id)
        with lock:
            workspace_dir = self._resolve_workspace(project_id)
            session = self._load_project_session(workspace_dir)

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_mode = request.mode or session.get("run_mode") or "step"
            start_node = request.start_node or "task_parsing"
            input_text = request.input_text if request.input_text is not None else session.get("input_text")
            if input_text is None:
                input_text = ""
            input_text = str(input_text)
            config = deepcopy(session.get("config") or load_global_config())

            session["status"] = "running"
            session["stage"] = "initialization"
            session["sub_stage"] = None
            session["run_mode"] = run_mode
            session["updated_at"] = datetime.now().isoformat()
            session["last_run_id"] = run_id
            if start_node == "task_parsing":
                session["input_text"] = input_text
            cancel_event = self._register_active_run(run_id, project_id)
            self._patch_project_session(
                workspace_dir,
                {
                    "status": session["status"],
                    "stage": session["stage"],
                    "sub_stage": session["sub_stage"],
                    "run_mode": session["run_mode"],
                    "updated_at": session["updated_at"],
                    "last_run_id": session["last_run_id"],
                    "input_text": session["input_text"],
                },
            )

            event_bus.publish(project_id, {
                "event": "run_started",
                "project_id": project_id,
                "run_id": run_id,
                "run_mode": run_mode,
                "start_node": start_node,
                "timestamp": datetime.now().isoformat(),
            })

            nodes: List[NodeResult] = []

            def _on_event(event: Dict[str, Any]) -> None:
                payload = dict(event)
                payload["project_id"] = project_id
                payload["run_id"] = run_id

                if payload.get("event") == "node_completed":
                    node = payload.get("node", "unknown")
                    delta = payload.get("delta") if isinstance(payload.get("delta"), dict) else {}
                    node_result = self._build_node_result(workspace_dir, node, delta)
                    nodes.append(node_result)
                    payload["node_result"] = node_result.model_dump()

                event_bus.publish(project_id, self._serialize(payload))

            researcher = AIResearcher(
                project_name=session.get("project_name", "research_project"),
                workspace_dir=workspace_dir,
                clear_workspace=False,
                model_preset=session.get("model_preset"),
            )

            try:
                final_state = researcher.run(
                    input_text=input_text,
                    run_id=run_id,
                    cancel_event=cancel_event,
                    start_node=start_node,
                    config=config,
                    mode=run_mode,
                    post_config=request.post_config,
                    event_callback=_on_event,
                )
            except RunCancelledError as e:
                latest_session = load_session_metadata(workspace_dir) or session
                interrupted_stage = str(latest_session.get("stage") or start_node or "unknown")
                interrupted_sub_stage = latest_session.get("sub_stage")
                interrupted_at = datetime.now().isoformat()
                self._patch_project_session(
                    workspace_dir,
                    {
                        "status": "interrupted",
                        "stage": interrupted_stage,
                        "sub_stage": interrupted_sub_stage,
                        "updated_at": interrupted_at,
                        "completed_at": interrupted_at,
                    },
                )
                event_bus.publish(project_id, {
                    "event": "run_cancelled",
                    "project_id": project_id,
                    "run_id": run_id,
                    "status": "interrupted",
                    "stage": interrupted_stage,
                    "sub_stage": interrupted_sub_stage,
                    "detail": str(e),
                    "timestamp": interrupted_at,
                })
                response = RunResponse(
                    project_id=project_id,
                    run_id=run_id,
                    run_mode=run_mode,
                    stage=interrupted_stage,
                    status="interrupted",
                    start_node=start_node,
                    nodes=nodes,
                    final_state={
                        "stage": interrupted_stage,
                        "sub_stage": interrupted_sub_stage,
                        "error": str(e),
                        "cancelled": True,
                    },
                )
                run_path = workspace_dir / "runs" / f"{run_id}.json"
                save_json(response.model_dump(), run_path)
                return response
            except Exception:
                self._patch_project_session(
                    workspace_dir,
                    {
                        "status": "failed",
                        "stage": "error",
                        "updated_at": datetime.now().isoformat(),
                    },
                )
                event_bus.publish(project_id, {
                    "event": "run_failed",
                    "project_id": project_id,
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                })
                raise
            finally:
                self._clear_active_run(run_id)

            status = "completed" if not final_state.get("error") else "failed"
            self._patch_project_session(
                workspace_dir,
                {
                    "status": status,
                    "stage": final_state.get("stage", "unknown"),
                    "updated_at": datetime.now().isoformat(),
                    "completed_at": datetime.now().isoformat(),
                },
            )

            response = RunResponse(
                project_id=project_id,
                run_id=run_id,
                run_mode=run_mode,
                stage=str(final_state.get("stage", "unknown")),
                status=status,
                start_node=start_node,
                nodes=nodes,
                final_state=self._serialize(final_state),
            )

            run_path = workspace_dir / "runs" / f"{run_id}.json"
            save_json(response.model_dump(), run_path)

            event_bus.publish(project_id, {
                "event": "run_completed",
                "project_id": project_id,
                "run_id": run_id,
                "status": status,
                "stage": response.stage,
                "timestamp": datetime.now().isoformat(),
            })

            return response

    def list_artifacts(self, project_id: str) -> ArtifactsResponse:
        workspace_dir = self._resolve_workspace(project_id)
        artifacts: List[ArtifactInfo] = []
        for path in workspace_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(workspace_dir)
            stat = path.stat()
            artifacts.append(
                ArtifactInfo(
                    path=str(rel),
                    size=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                )
            )
        artifacts.sort(key=lambda x: x.modified_at, reverse=True)
        return ArtifactsResponse(project_id=project_id, artifacts=artifacts)

    def get_artifact_content(self, project_id: str, artifact_path: str) -> ArtifactContentResponse:
        workspace_dir = self._resolve_workspace(project_id).resolve()
        target_path = (workspace_dir / artifact_path).resolve()
        if not str(target_path).startswith(str(workspace_dir)):
            raise PermissionError("Invalid artifact path")
        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        if self._is_text_suffix(target_path.suffix):
            content = target_path.read_text(encoding="utf-8")
        else:
            content = f"[binary file] {target_path.name}"

        return ArtifactContentResponse(
            project_id=project_id,
            path=str(target_path.relative_to(workspace_dir)),
            content=content,
        )

    def update_artifact_content(
        self,
        project_id: str,
        artifact_path: str,
        request: ArtifactUpdateRequest,
    ) -> ArtifactUpdateResponse:
        lock = self._get_project_lock(project_id)
        with lock:
            workspace_dir = self._resolve_workspace(project_id).resolve()
            target_path = (workspace_dir / artifact_path).resolve()
            if not str(target_path).startswith(str(workspace_dir)):
                raise PermissionError("Invalid artifact path")
            if not target_path.exists() or not target_path.is_file():
                raise FileNotFoundError(f"Artifact not found: {artifact_path}")

            if not self._is_text_suffix(target_path.suffix):
                raise ValueError(f"Artifact type is not editable: {target_path.suffix}")

            target_path.write_text(request.content, encoding="utf-8")
            stat = target_path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime).isoformat()

            # Update project session heartbeat timestamp for UI freshness.
            self._patch_project_session(
                workspace_dir,
                {"updated_at": datetime.now().isoformat()},
            )

            return ArtifactUpdateResponse(
                project_id=project_id,
                path=str(target_path.relative_to(workspace_dir)),
                size=stat.st_size,
                updated_at=updated_at,
            )

    def list_files(self, project_id: str) -> FilesResponse:
        workspace_dir = self._resolve_workspace(project_id)
        files: List[FileInfo] = []
        for path in workspace_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(workspace_dir)
            stat = path.stat()
            files.append(
                FileInfo(
                    path=str(rel),
                    size=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    is_text=self._is_text_suffix(path.suffix),
                )
            )
        files.sort(key=lambda x: x.modified_at, reverse=True)
        return FilesResponse(project_id=project_id, files=files)

    def build_project_zip(self, project_id: str) -> Path:
        workspace_dir = self._resolve_workspace(project_id).resolve()
        fd, tmp_zip_path = tempfile.mkstemp(prefix=f"{project_id}_", suffix=".zip")
        os.close(fd)
        zip_path = Path(tmp_zip_path)

        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in workspace_dir.rglob("*"):
                if not path.is_file():
                    continue
                archive.write(path, arcname=str(path.relative_to(workspace_dir)))

        return zip_path

    def get_file_content(self, project_id: str, file_path: str) -> FileContentResponse:
        workspace_dir = self._resolve_workspace(project_id).resolve()
        target_path = self._resolve_project_file_path(project_id, file_path)

        if self._is_text_suffix(target_path.suffix):
            return FileContentResponse(
                project_id=project_id,
                path=str(target_path.relative_to(workspace_dir)),
                content=target_path.read_text(encoding="utf-8"),
                is_text=True,
                encoding="utf-8",
            )

        return FileContentResponse(
            project_id=project_id,
            path=str(target_path.relative_to(workspace_dir)),
            content=None,
            is_text=False,
            encoding="base64",
        )

    def get_file_download_path(self, project_id: str, file_path: str) -> Path:
        return self._resolve_project_file_path(project_id, file_path)

    def upsert_project_file(
        self,
        project_id: str,
        file_path: str,
        request: FileUpsertRequest,
    ) -> FileWriteResponse:
        lock = self._get_project_lock(project_id)
        with lock:
            workspace_dir = self._resolve_workspace(project_id).resolve()
            target_path = (workspace_dir / file_path).resolve()
            if not str(target_path).startswith(str(workspace_dir)):
                raise PermissionError("Invalid file path")

            existed = target_path.exists()
            if existed and not request.overwrite:
                raise FileExistsError(f"File already exists: {file_path}")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            if request.encoding == "utf-8":
                target_path.write_text(request.content, encoding="utf-8")
            else:
                try:
                    payload = base64.b64decode(request.content, validate=True)
                except Exception as exc:
                    raise ValueError("Invalid base64 content") from exc
                target_path.write_bytes(payload)

            stat = target_path.stat()
            updated_at = datetime.fromtimestamp(stat.st_mtime).isoformat()

            self._patch_project_session(
                workspace_dir,
                {"updated_at": datetime.now().isoformat()},
            )

            return FileWriteResponse(
                project_id=project_id,
                path=str(target_path.relative_to(workspace_dir)),
                size=stat.st_size,
                updated_at=updated_at,
                created=not existed,
            )

    def list_node_history(self, project_id: str, node_name: str) -> NodeHistoryResponse:
        workspace_dir = self._resolve_workspace(project_id)
        history_dir = workspace_dir / "history" / node_name
        files = []
        if history_dir.exists():
            files = [str(path.relative_to(workspace_dir)) for path in sorted(history_dir.glob("*.json"))]
        return NodeHistoryResponse(project_id=project_id, node=node_name, files=files)

    def latest_node_result(self, project_id: str, node_name: str) -> NodeResult:
        workspace_dir = self._resolve_workspace(project_id)
        session = self._reconcile_project_session(
            project_id,
            workspace_dir,
            load_session_metadata(workspace_dir) or {},
        )

        if (
            session.get("status") == "running"
            and session.get("stage") == node_name
        ):
            return self._build_node_result(
                workspace_dir,
                node_name,
                {
                    "stage": node_name,
                    "sub_stage": session.get("sub_stage"),
                    "status": "running",
                },
            )

        # TODO
        runs_dir = workspace_dir / "runs"
        if runs_dir.exists():
            run_files = sorted(runs_dir.glob("*.json"))
            if run_files:
                latest_run = load_json(run_files[-1]) or {}
                for node_data in reversed(latest_run.get("nodes", [])):
                    if node_data.get("node") == node_name:
                        return NodeResult(**node_data)

        return self._build_node_result(workspace_dir, node_name, {"stage": node_name})

    def get_logs(self, project_id: str, tail_lines: int = 200) -> LogsResponse:
        workspace_dir = self._resolve_workspace(project_id)
        return LogsResponse(project_id=project_id, lines=self._read_log_tail(workspace_dir, tail_lines=tail_lines))
