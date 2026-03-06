from __future__ import annotations

from queue import SimpleQueue
from threading import Lock
from typing import Any, Dict, Tuple
from uuid import uuid4


class ProjectEventBus:
    """Thread-safe in-memory pub/sub for project-scoped events."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._queues: Dict[str, Dict[str, SimpleQueue]] = {}

    def subscribe(self, project_id: str) -> Tuple[str, SimpleQueue]:
        token = uuid4().hex
        queue: SimpleQueue = SimpleQueue()
        with self._lock:
            self._queues.setdefault(project_id, {})[token] = queue
        return token, queue

    def unsubscribe(self, project_id: str, token: str) -> None:
        with self._lock:
            subscribers = self._queues.get(project_id)
            if not subscribers:
                return
            subscribers.pop(token, None)
            if not subscribers:
                self._queues.pop(project_id, None)

    def publish(self, project_id: str, event: Dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._queues.get(project_id, {}).values())
        for queue in subscribers:
            queue.put(event)
