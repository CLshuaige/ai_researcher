from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

# node name list
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


class ProjectCreateRequest(BaseModel):
    project_name: str = Field(default="research_project")
    input_text: str = Field(default=None)
    config_path: Optional[str] = Field(default=None)
    mode: Literal["step", "auto"] = Field(default="step")
    model_preset: Optional[str] = Field(default=None)


class ProjectCreateResponse(BaseModel):
    project_id: str # 展示给用户
    project_name: str
    workspace_dir: str
    run_mode: str
    available_nodes: List[str]


class ConfigUpdateRequest(BaseModel):
    """
    用于更新参数列表,格式参照config.json.
    例如页面一修改human_in_the_loop为true,则config_patch为
    {
        "researcher": {
            "task_parsing": {
                "human_in_the_loop": true
            }
        }
    }
    """
    config_patch: Dict[str, Any] = Field(default_factory=dict) 


class ConfigUpdateResponse(BaseModel):
    project_id: str
    applied_patch: Dict[str, Any]
    researcher_config: Dict[str, Any] # 根据返回的配置文件来渲染前端界面。例如页面一：根据human_in_the_loop字段来显示滑动按钮的状态

class RunRequest(BaseModel):
    mode: Optional[Literal["step", "auto"]] = Field(default=None)
    start_node: Optional[str] = Field(default=None) # 命名规范按照 本文件中变量：NODE_ORDER
    input_text: Optional[str] = Field(default=None) # 仅“任务解析”页面使用
    post_config: Dict[str, Any] = Field(default_factory=dict) # 空置


class NodeProcess(BaseModel):
    history_path: Optional[str] = None
    message_count: int = 0
    log_tail: List[str] = Field(default_factory=list)


class NodeResult(BaseModel):
    node: str
    stage: str
    status: str # 用于前端展示当前状态，检测到status为completed时展示artifacts字段的内容。
    next_node: Optional[str] = None 
    process: NodeProcess
    output: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list) # 以按钮的形式呈现每个artifact名称。点击后触发api-10。api-10中的{file_path:path}设置为artifact名称。
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
    projects: List[ProjectStatusResponse] = Field(default_factory=list) # 呈现当前已有的projects
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
    content: Optional[str] = None # 以markdown的形式渲染在页面中
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
    created: bool # 显示上传是否成功


class NodeHistoryResponse(BaseModel):
    project_id: str
    node: str
    files: List[str]


class LogsResponse(BaseModel):
    project_id: str
    lines: List[str]

class UserInputRequest(BaseModel):
    request_id: str
    value: str