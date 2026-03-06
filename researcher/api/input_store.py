import threading

# 在 InputResponseStore 中新增同步等待方法
class InputResponseStore:
    def __init__(self):
        self.pending = {}

    def create(self, request_id):
        evt = threading.Event()
        self.pending[request_id] = {"event": evt, "value": None}
        return evt

    def resolve(self, request_id, value):
        fut = self.pending.pop(request_id, None)
        if fut:
            fut["value"] = value
            fut["event"].set()

    def wait_for_input(self, request_id):
        fut = self.pending.get(request_id)
        if fut:
            fut["event"].wait()  # 阻塞等待
            return fut["value"]
        return None