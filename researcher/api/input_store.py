import threading
import time
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

    def wait_for_input(self, request_id: str, timeout: int = 600, poll_interval: float = 1, cancel_check=None):
        with self.lock:
            fut = self.pending.get(request_id)

        if not fut:
            return None

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self.lock:
                    self.pending.pop(request_id, None)
                raise TimeoutError(f"user input timeout: {request_id}")

            if cancel_check and cancel_check():
                with self.lock:
                    self.pending.pop(request_id, None)
                raise InterruptedError(f"user input cancelled: {request_id}")

            wait_time = min(poll_interval, remaining)
            if fut["event"].wait(wait_time):
                with self.lock:
                    self.pending.pop(request_id, None)
                return fut["value"]

    def pending_ids(self):
        with self.lock:
            return list(self.pending.keys())
