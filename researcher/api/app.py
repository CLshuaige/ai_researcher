from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask

from researcher.api.events import ProjectEventBus
from researcher.api.schemas import (
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    FileContentResponse,
    FilesResponse,
    FileUpsertRequest,
    FileWriteResponse,
    LogsResponse,
    NodeHistoryResponse,
    NodeResult,
    ProjectListResponse,
    ProjectCreateRequest,
    ProjectCreateResponse,
    ProjectStatusResponse,
    RunRequest,
    RunResponse,
    UserInputRequest,
)
from researcher.api.service import APIProjectService
from researcher.api.input_store import InputResponseStore


app = FastAPI(title="AI Researcher API", version="0.1.0")
input_store = InputResponseStore()
service = APIProjectService()
event_bus = ProjectEventBus()
app.state.event_bus = event_bus


def _cleanup_file(path: str) -> None:
    Path(path).unlink(missing_ok=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# api-1
@app.get("/health") 
def health() -> dict:
    return {"status": "ok"}

# api-2
@app.post("/api/v1/projects", response_model=ProjectCreateResponse)  
async def create_project(request: ProjectCreateRequest) -> ProjectCreateResponse:
    try:
        return await run_in_threadpool(service.create_project, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-3
@app.get("/api/v1/projects", response_model=ProjectListResponse) 
async def list_projects() -> ProjectListResponse:
    try:
        return await run_in_threadpool(service.list_projects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-4
@app.get("/api/v1/projects/latest", response_model=ProjectStatusResponse) 
async def latest_project() -> ProjectStatusResponse:
    try:
        return await run_in_threadpool(service.latest_project)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-5
@app.post("/api/v1/projects/{project_id}/config", response_model=ConfigUpdateResponse)
async def update_project_config(project_id: str, request: ConfigUpdateRequest) -> ConfigUpdateResponse:
    try:
        return await run_in_threadpool(service.update_project_config, project_id, request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-6
@app.post("/api/v1/projects/{project_id}/runs", response_model=RunResponse)
async def run_project(project_id: str, request: RunRequest) -> RunResponse:
    try:
        return await run_in_threadpool(service.run_project, project_id, request, event_bus)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-7：暂时不用
@app.get("/api/v1/projects/{project_id}", response_model=ProjectStatusResponse)
async def get_project_status(project_id: str) -> ProjectStatusResponse:
    try:
        return await run_in_threadpool(service.get_project_status, project_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-8
@app.get("/api/v1/projects/{project_id}/nodes/{node_name}/latest", response_model=NodeResult)
async def latest_node_result(project_id: str, node_name: str) -> NodeResult:
    try:
        return await run_in_threadpool(service.latest_node_result, project_id, node_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-9
@app.get("/api/v1/projects/{project_id}/files", response_model=FilesResponse)
async def list_files(project_id: str, download: bool = Query(default=False)) -> Any:
    try:
        if download:
            zip_path = await run_in_threadpool(service.build_project_zip, project_id)
            return FileResponse(
                path=str(zip_path),
                media_type="application/zip",
                filename=f"{project_id}.zip",
                background=BackgroundTask(_cleanup_file, str(zip_path)),
            )
        return await run_in_threadpool(service.list_files, project_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-10
@app.get("/api/v1/projects/{project_id}/files/{file_path:path}", response_model=FileContentResponse)
async def get_file_content(project_id: str, file_path: str, download: bool = Query(default=False)) -> Any:
    try:
        if download:
            target_path = await run_in_threadpool(service.get_file_download_path, project_id, file_path)
            media_type, _ = mimetypes.guess_type(str(target_path))
            return FileResponse(
                path=str(target_path),
                media_type=media_type or "application/octet-stream",
                filename=target_path.name,
            )
        return await run_in_threadpool(service.get_file_content, project_id, file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-11
@app.put("/api/v1/projects/{project_id}/files/{file_path:path}", response_model=FileWriteResponse)
async def upsert_project_file(
    project_id: str,
    file_path: str,
    request: FileUpsertRequest,
) -> FileWriteResponse:
    try:
        return await run_in_threadpool(service.upsert_project_file, project_id, file_path, request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-12
@app.get("/api/v1/projects/{project_id}/history/{node_name}", response_model=NodeHistoryResponse)
async def list_node_history(project_id: str, node_name: str) -> NodeHistoryResponse:
    try:
        return await run_in_threadpool(service.list_node_history, project_id, node_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-13
@app.get("/api/v1/projects/{project_id}/logs", response_model=LogsResponse)
async def get_logs(project_id: str, tail_lines: int = Query(default=200, ge=1, le=2000)) -> LogsResponse:
    try:
        return await run_in_threadpool(service.get_logs, project_id, tail_lines)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# api-14
@app.websocket("/api/v1/projects/{project_id}/events")
async def project_events(project_id: str, websocket: WebSocket):

    await websocket.accept()

    token, queue = event_bus.subscribe(project_id)

    try:
        while True:

            try:
                event = await asyncio.wait_for(
                    queue.get(),
                    timeout=15
                )

                await websocket.send_json(event)

            except asyncio.TimeoutError:

                await websocket.send_json({
                    "event": "heartbeat",
                    "project_id": project_id,
                })

    except WebSocketDisconnect:
        pass

    finally:
        event_bus.unsubscribe(project_id, token)


# api-15
@app.post("/api/v1/projects/{project_id}/input")
async def user_input(project_id: str, request: UserInputRequest):

    if not input_store.resolve(request.request_id, request.value):
        raise HTTPException(
            status_code=404,
            detail="request_id not found or already resolved"
        )

    return {
        "status": "ok",
        "project_id": project_id,
        "request_id": request.request_id
    }
