from __future__ import annotations

import os
import socket
import stat
import subprocess
import textwrap
import time
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "vllm"


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_companion_script_lifecycle_and_management(tmp_path: Path) -> None:
    project = tmp_path / "project"
    python_dir = project / ".venv" / "bin"
    python_dir.mkdir(parents=True)
    (project / "models.toml").write_text(
        '["model-a"]\nmodel_path = "org/model-a"\ndefault = true\n',
        encoding="utf-8",
    )

    # The filename contains "vllm-gateway" so the companion script's
    # pid_matches_gateway recognizes this fake process.
    fake_server = tmp_path / "vllm-gateway-fake.py"
    fake_server.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import signal
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            class Handler(BaseHTTPRequestHandler):
                def _write(self, status, payload):
                    body = json.dumps(payload).encode()
                    self.send_response(status)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_GET(self):
                    if self.path == "/health":
                        self._write(200, {"status": "ok", "loaded_models": ["model-a"]})
                    elif self.path == "/ready":
                        self._write(200, {"status": "ok"})
                    elif self.path == "/admin/models":
                        self._write(200, {"loaded_models": ["model-a"]})
                    else:
                        self._write(404, {"error": "not found"})

                def do_POST(self):
                    if self.path.endswith("/load") or self.path.endswith("/unload"):
                        self._write(200, {"ok": True, "path": self.path})
                    else:
                        self._write(404, {"error": "not found"})

                def log_message(self, *_):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", int(os.environ["VLLM_SERVICE_PORT"])), Handler)
            signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
            server.serve_forever()
            """
        ),
        encoding="utf-8",
    )

    fake_python = python_dir / "python"
    fake_python.write_text(
        '#!/bin/sh\nexec python3 "$FAKE_SERVER"\n',
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

    port = free_port()
    env = {
        **os.environ,
        "VLLM_PROJECT_DIR": str(project),
        "VLLM_CONFIG_FILE": str(project / "models.toml"),
        "VLLM_SERVICE_PORT": str(port),
        "VLLM_PID_FILE": str(tmp_path / "gateway.pid"),
        "VLLM_LOG_FILE": str(tmp_path / "gateway.log"),
        "VLLM_GATEWAY_START_TIMEOUT": "5",
        "VLLM_STOP_TIMEOUT": "5",
        "FAKE_SERVER": str(fake_server),
    }

    def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SCRIPT), *args],
            env=env,
            text=True,
            capture_output=True,
            timeout=15,
            check=check,
        )

    try:
        started = run("start")
        assert "Started vLLM gateway" in started.stdout

        status = run("status")
        assert "Readiness: ready" in status.stdout
        assert '"loaded_models": ["model-a"]' in status.stdout

        models = run("models")
        assert '"loaded_models": ["model-a"]' in models.stdout
        assert '"ok": true' in run("load", "model-a").stdout.lower()
        assert '"ok": true' in run("unload", "model-a").stdout.lower()

        restarted = run("restart")
        assert "vLLM gateway stopped" in restarted.stdout
        assert "Started vLLM gateway" in restarted.stdout
    finally:
        run("stop", check=False)

    stopped = run("status", check=False)
    assert stopped.returncode == 3
    assert "not running" in stopped.stdout


def test_start_refuses_foreign_public_port_without_signalling_owner(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = project / "models.toml"
    config.write_text('["model-a"]\nmodel_path = "org/a"\n', encoding="utf-8")
    port = free_port()
    owner = subprocess.Popen(
        ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            probe = socket.socket()
            try:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            finally:
                probe.close()
            time.sleep(0.02)
        result = subprocess.run(
            [str(SCRIPT), "start"],
            env={
                **os.environ,
                "VLLM_PROJECT_DIR": str(project),
                "VLLM_CONFIG_FILE": str(config),
                "VLLM_SERVICE_PORT": str(port),
                "VLLM_PID_FILE": str(tmp_path / "gateway.pid"),
                "VLLM_LOG_FILE": str(tmp_path / "gateway.log"),
            },
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert "already in use" in result.stderr
        assert owner.poll() is None
    finally:
        owner.terminate()
        owner.wait(timeout=5)


def test_script_does_not_scan_or_kill_backend_processes() -> None:
    content = SCRIPT.read_text(encoding="utf-8")
    assert "fuser" not in content
    assert "pgrep" not in content
    assert "backend_pids" not in content
