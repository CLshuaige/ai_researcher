from __future__ import annotations

import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.concurrency import run_in_threadpool

from researcher.api.events import ProjectEventBus
from researcher.api.schemas import (
    ArtifactContentResponse,
    ArtifactsResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    LogsResponse,
    NodeHistoryResponse,
    NodeResult,
    ProjectListResponse,
    ProjectCreateRequest,
    ProjectCreateResponse,
    ProjectStatusResponse,
    RunRequest,
    RunResponse,
)
from researcher.api.service import APIProjectService


app = FastAPI(title="AI Researcher API", version="0.1.0")
service = APIProjectService()
event_bus = ProjectEventBus()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/projects", response_model=ProjectCreateResponse)
async def create_project(request: ProjectCreateRequest) -> ProjectCreateResponse:
    try:
        return await run_in_threadpool(service.create_project, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects", response_model=ProjectListResponse)
async def list_projects() -> ProjectListResponse:
    try:
        return await run_in_threadpool(service.list_projects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/latest", response_model=ProjectStatusResponse)
async def latest_project() -> ProjectStatusResponse:
    try:
        return await run_in_threadpool(service.latest_project)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/projects/{project_id}/config", response_model=ConfigUpdateResponse)
async def update_project_config(project_id: str, request: ConfigUpdateRequest) -> ConfigUpdateResponse:
    try:
        return await run_in_threadpool(service.update_project_config, project_id, request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/projects/{project_id}/runs", response_model=RunResponse)
async def run_project(project_id: str, request: RunRequest) -> RunResponse:
    try:
        return await run_in_threadpool(service.run_project, project_id, request, event_bus)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/{project_id}", response_model=ProjectStatusResponse)
async def get_project_status(project_id: str) -> ProjectStatusResponse:
    try:
        return await run_in_threadpool(service.get_project_status, project_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/{project_id}/nodes/{node_name}/latest", response_model=NodeResult)
async def latest_node_result(project_id: str, node_name: str) -> NodeResult:
    try:
        return await run_in_threadpool(service.latest_node_result, project_id, node_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/{project_id}/artifacts", response_model=ArtifactsResponse)
async def list_artifacts(project_id: str) -> ArtifactsResponse:
    try:
        return await run_in_threadpool(service.list_artifacts, project_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/{project_id}/artifacts/{artifact_path:path}", response_model=ArtifactContentResponse)
async def get_artifact_content(project_id: str, artifact_path: str) -> ArtifactContentResponse:
    try:
        return await run_in_threadpool(service.get_artifact_content, project_id, artifact_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/{project_id}/history/{node_name}", response_model=NodeHistoryResponse)
async def list_node_history(project_id: str, node_name: str) -> NodeHistoryResponse:
    try:
        return await run_in_threadpool(service.list_node_history, project_id, node_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/{project_id}/logs", response_model=LogsResponse)
async def get_logs(project_id: str, tail_lines: int = Query(default=200, ge=1, le=2000)) -> LogsResponse:
    try:
        return await run_in_threadpool(service.get_logs, project_id, tail_lines)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/v1/projects/{project_id}/events")
async def project_events(project_id: str, websocket: WebSocket) -> None:
    await websocket.accept()
    token, queue = event_bus.subscribe(project_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(asyncio.to_thread(queue.get), timeout=15.0)
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
