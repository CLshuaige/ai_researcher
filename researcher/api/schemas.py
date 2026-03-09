from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class ProjectCreateRequest(BaseModel):
    project_name: str = Field(default="research_project")
    input_text: str = Field(default=None)
    config_path: Optional[str] = Field(default=None)
    mode: Literal["step", "auto"] = Field(default="step")
    model_preset: Optional[str] = Field(default=None)


class ProjectCreateResponse(BaseModel):
    project_id: str
    project_name: str
    workspace_dir: str
    run_mode: str
    available_nodes: List[str]


class ConfigUpdateRequest(BaseModel):
    config_patch: Dict[str, Any] = Field(default_factory=dict)


class ConfigUpdateResponse(BaseModel):
    project_id: str
    applied_patch: Dict[str, Any]
    researcher_config: Dict[str, Any]

class RunRequest(BaseModel):
    mode: Optional[Literal["step", "auto"]] = Field(default=None)
    start_node: Optional[str] = Field(default=None)
    input_text: Optional[str] = Field(default=None)
    post_config: Dict[str, Any] = Field(default_factory=dict)


class NodeProcess(BaseModel):
    history_path: Optional[str] = None
    message_count: int = 0
    log_tail: List[str] = Field(default_factory=list)


class NodeResult(BaseModel):
    node: str
    stage: str
    next_node: Optional[str] = None
    process: NodeProcess
    output: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    opencode: Optional[Dict[str, Any]] = None


class RunResponse(BaseModel):
    project_id: str
    run_id: str
    run_mode: str
    stage: str
    status: str
    start_node: str
    nodes: List[NodeResult] = Field(default_factory=list)
    final_state: Dict[str, Any] = Field(default_factory=dict)


class ProjectStatusResponse(BaseModel):
    project_id: str
    project_name: str
    status: str
    stage: str
    run_mode: str
    workspace_dir: str
    updated_at: Optional[str] = None


class ProjectListResponse(BaseModel):
    projects: List[ProjectStatusResponse] = Field(default_factory=list)
    total: int = 0


class ArtifactInfo(BaseModel):
    path: str
    size: int
    modified_at: str


class ArtifactsResponse(BaseModel):
    project_id: str
    artifacts: List[ArtifactInfo]


class ArtifactContentResponse(BaseModel):
    project_id: str
    path: str
    content: str


class ArtifactUpdateRequest(BaseModel):
    content: str


class ArtifactUpdateResponse(BaseModel):
    project_id: str
    path: str
    size: int
    updated_at: str


class FileInfo(BaseModel):
    path: str
    size: int
    modified_at: str
    is_text: bool


class FilesResponse(BaseModel):
    project_id: str
    files: List[FileInfo]


class FileContentResponse(BaseModel):
    project_id: str
    path: str
    content: Optional[str] = None
    is_text: bool
    encoding: Literal["utf-8", "base64"]


class FileUpsertRequest(BaseModel):
    content: str
    encoding: Literal["utf-8", "base64"] = Field(default="utf-8")
    overwrite: bool = Field(default=False)


class FileWriteResponse(BaseModel):
    project_id: str
    path: str
    size: int
    updated_at: str
    created: bool


class NodeHistoryResponse(BaseModel):
    project_id: str
    node: str
    files: List[str]


class LogsResponse(BaseModel):
    project_id: str
    lines: List[str]
