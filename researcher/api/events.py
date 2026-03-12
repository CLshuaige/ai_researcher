import asyncio
from threading import Lock
from typing import Any, Dict, Tuple
from uuid import uuid4


class ProjectEventBus:

    def __init__(self):
        self._lock = Lock()
        self._queues: Dict[str, Dict[str, asyncio.Queue]] = {}

    def subscribe(self, project_id: str) -> Tuple[str, asyncio.Queue]:

        token = uuid4().hex
        queue: asyncio.Queue = asyncio.Queue()

        with self._lock:
            self._queues.setdefault(project_id, {})[token] = queue

        return token, queue

    def unsubscribe(self, project_id: str, token: str):

        with self._lock:
            subs = self._queues.get(project_id)

            if not subs:
                return

            subs.pop(token, None)

            if not subs:
                self._queues.pop(project_id, None)

    def publish(self, project_id: str, event: Dict[str, Any]):

        with self._lock:
            subscribers = list(self._queues.get(project_id, {}).values())

        for q in subscribers:
            q.put_nowait(event)