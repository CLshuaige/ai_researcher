from __future__ import annotations

import socket
import subprocess
import time

import uvicorn

from researcher.api.app import app


HOST = "0.0.0.0"
PORT = 8001


if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        port_in_use = sock.connect_ex(("127.0.0.1", PORT)) == 0

    if port_in_use:
        out = subprocess.check_output(["lsof", "-t", f"-iTCP:{PORT}", "-sTCP:LISTEN"], text=True)
        pids = sorted({int(x.strip()) for x in out.splitlines() if x.strip().isdigit()})
        for pid in pids:
            subprocess.run(["kill", "-15", str(pid)], check=False)
        time.sleep(1.0)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            still_in_use = sock.connect_ex(("127.0.0.1", PORT)) == 0

        if still_in_use:
            out = subprocess.check_output(["lsof", "-t", f"-iTCP:{PORT}", "-sTCP:LISTEN"], text=True)
            pids = sorted({int(x.strip()) for x in out.splitlines() if x.strip().isdigit()})
            for pid in pids:
                subprocess.run(["kill", "-9", str(pid)], check=False)
            time.sleep(0.5)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", PORT)) == 0:
                raise RuntimeError(f"port {PORT} still occupied")

    uvicorn.run("api_main:app", host=HOST, port=PORT, reload=False)
