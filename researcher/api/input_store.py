import threading
from typing import Dict


class InputResponseStore:
    def __init__(self):
        self.pending: Dict[str, dict] = {}
        self.lock = threading.Lock()

    def create(self, request_id: str):
        evt = threading.Event()

        with self.lock:
            self.pending[request_id] = {
                "event": evt,
                "value": None,
            }

    def resolve(self, request_id: str, value: str):
        with self.lock:
            fut = self.pending.get(request_id)

        if not fut:
            return False

        fut["value"] = value
        fut["event"].set()

        return True

    def wait_for_input(self, request_id: str, timeout: int = 600):
        with self.lock:
            fut = self.pending.get(request_id)

        if not fut:
            return None

        ok = fut["event"].wait(timeout)

        with self.lock:
            self.pending.pop(request_id, None)

        if not ok:
            raise TimeoutError(f"user input timeout: {request_id}")

        return fut["value"]

    def pending_ids(self):
        with self.lock:
            return list(self.pending.keys())