from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from public_connection_models import PublicAccessState


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_path_prefix(path_prefix: str) -> str:
    normalized = "/" + str(path_prefix or "").strip().strip("/")
    if normalized == "/":
        return "/public"
    return normalized


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


class PublicAccessController:
    def __init__(
        self,
        root: Path,
        *,
        local_target_url: str,
        https_port: int = 8443,
        path_prefix: str = "/public",
        sync_interval_seconds: float = 30.0,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "server_public_access.json"
        self.local_target_url = str(local_target_url)
        self.https_port = int(https_port)
        self.path_prefix = _normalize_path_prefix(path_prefix)
        self.sync_interval_seconds = max(5.0, float(sync_interval_seconds))
        self._command_runner = command_runner or self._default_command_runner
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        if not self.path.exists():
            self._write_state_locked(
                PublicAccessState(
                    desired_enabled=False,
                    active=False,
                    api_key="",
                    https_port=self.https_port,
                    path_prefix=self.path_prefix,
                    local_target_url=self.local_target_url,
                )
            )

    def start(self) -> None:
        with self._lock:
            self._synchronize_locked(self._read_state_locked())
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._sync_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def get_state(self) -> PublicAccessState:
        with self._lock:
            state = self._read_state_locked()
            return self._refresh_runtime_state_locked(state)

    def get_cached_state(self) -> PublicAccessState:
        with self._lock:
            return self._read_state_locked()

    def update_state(self, *, enabled: bool, api_key: str | None = None) -> PublicAccessState:
        with self._lock:
            state = self._read_state_locked()
            state.desired_enabled = bool(enabled)
            if api_key is not None:
                state.api_key = str(api_key).strip()
            state.updated_at = _utc_now()
            self._write_state_locked(state)
            return self._synchronize_locked(state)

    def sync_now(self) -> PublicAccessState:
        with self._lock:
            state = self._read_state_locked()
            return self._synchronize_locked(state)

    def _sync_loop(self) -> None:
        while not self._stop_event.wait(self.sync_interval_seconds):
            try:
                self.sync_now()
            except Exception:
                continue

    def _synchronize_locked(self, state: PublicAccessState) -> PublicAccessState:
        refreshed = self._refresh_runtime_state_locked(state)
        actual_payload = self._tailscale_funnel_status_locked()
        active = self._is_expected_funnel_active(actual_payload)
        has_other_config = self._has_conflicting_funnel_config(actual_payload)
        if refreshed.desired_enabled:
            if has_other_config:
                refreshed.last_error = (
                    "Funnel is already using this port/path for something else; refusing to overwrite it."
                )
            elif not active:
                result = self._run_command_locked(
                    [
                        "tailscale",
                        "funnel",
                        "--bg",
                        "--yes",
                        f"--https={refreshed.https_port}",
                        f"--set-path={refreshed.path_prefix}",
                        refreshed.local_target_url,
                    ]
                )
                if result.returncode != 0:
                    refreshed.last_error = self._format_command_error(result)
                    refreshed.operator_required = self._operator_required(result)
                else:
                    refreshed.last_error = None
                    refreshed.operator_required = False
        else:
            if active and not has_other_config:
                result = self._run_command_locked(
                    [
                        "tailscale",
                        "funnel",
                        f"--https={refreshed.https_port}",
                        f"--set-path={refreshed.path_prefix}",
                        "off",
                    ]
                )
                if result.returncode != 0:
                    refreshed.last_error = self._format_command_error(result)
                    refreshed.operator_required = self._operator_required(result)
                else:
                    refreshed.last_error = None
                    refreshed.operator_required = False
            elif has_other_config:
                refreshed.last_error = (
                    "Funnel is using this port/path for something else; not changing it from this app."
                )
        return self._refresh_runtime_state_locked(refreshed)

    def _refresh_runtime_state_locked(self, state: PublicAccessState) -> PublicAccessState:
        actual_payload = self._tailscale_funnel_status_locked()
        dns_name = self._tailscale_dns_name_locked()
        state.active = self._is_expected_funnel_active(actual_payload)
        if state.active:
            state.last_error = None
            state.operator_required = False
        state.dns_name = dns_name
        state.public_url = self._public_url_for_dns_name(dns_name)
        state.https_port = self.https_port
        state.path_prefix = self.path_prefix
        state.local_target_url = self.local_target_url
        state.last_synced_at = _utc_now()
        self._write_state_locked(state)
        return state

    def _public_url_for_dns_name(self, dns_name: str | None) -> str | None:
        if not dns_name:
            return None
        base = dns_name.rstrip(".")
        return f"https://{base}:{self.https_port}{self.path_prefix}/"

    def _operator_required(self, result: subprocess.CompletedProcess[str]) -> bool:
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        lowered = combined.lower()
        return "use 'sudo tailscale funnel" in lowered or "tailscale set --operator" in lowered

    def _format_command_error(self, result: subprocess.CompletedProcess[str]) -> str:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        joined = "\n".join(part for part in (stdout, stderr) if part)
        return joined or f"Command failed with exit code {result.returncode}"

    def _default_command_runner(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, text=True, capture_output=True, check=False)

    def _run_command_locked(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        return self._command_runner([str(part) for part in argv])

    def _tailscale_funnel_status_locked(self) -> dict[str, Any]:
        result = self._run_command_locked(["tailscale", "funnel", "status", "--json"])
        if result.returncode != 0:
            return {}
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _tailscale_dns_name_locked(self) -> str | None:
        result = self._run_command_locked(["tailscale", "status", "--json"])
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return None
        self_payload = payload.get("Self")
        if not isinstance(self_payload, dict):
            return None
        dns_name = self_payload.get("DNSName")
        if not isinstance(dns_name, str) or not dns_name.strip():
            return None
        return dns_name.rstrip(".")

    def _is_expected_funnel_active(self, payload: dict[str, Any]) -> bool:
        handler = self._expected_handler(payload)
        if not isinstance(handler, dict):
            return False
        return str(handler.get("Proxy") or "").rstrip("/") == self.local_target_url.rstrip("/")

    def _has_conflicting_funnel_config(self, payload: dict[str, Any]) -> bool:
        if not payload:
            return False
        https_payload = payload.get("https")
        if isinstance(https_payload, dict):
            config = https_payload.get(str(self.https_port))
            if config is None:
                return False
            if not isinstance(config, dict):
                return True
            path_value = str(config.get("path") or "")
            target = str(config.get("target") or "").rstrip("/")
            if path_value != self.path_prefix:
                return True
            return target != self.local_target_url.rstrip("/")
        web = payload.get("Web", {})
        if not isinstance(web, dict):
            return False
        for host_port, config in web.items():
            if not str(host_port).endswith(f":{self.https_port}"):
                continue
            handlers = config.get("Handlers", {})
            if not isinstance(handlers, dict):
                continue
            if self.path_prefix not in handlers:
                continue
            handler = handlers.get(self.path_prefix)
            if not isinstance(handler, dict):
                return True
            proxy = str(handler.get("Proxy") or "").rstrip("/")
            if proxy != self.local_target_url.rstrip("/"):
                return True
        return False

    def _expected_handler(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not payload:
            return None
        https_payload = payload.get("https")
        if isinstance(https_payload, dict):
            config = https_payload.get(str(self.https_port))
            if isinstance(config, dict):
                return {"Proxy": config.get("target"), "Path": config.get("path")}
        web = payload.get("Web", {})
        if not isinstance(web, dict):
            return None
        for host_port, config in web.items():
            if not str(host_port).endswith(f":{self.https_port}"):
                continue
            handlers = config.get("Handlers", {})
            if not isinstance(handlers, dict):
                continue
            handler = handlers.get(self.path_prefix)
            if isinstance(handler, dict):
                return handler
        return None

    def _read_state_locked(self) -> PublicAccessState:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return PublicAccessState.model_validate(payload)

    def _write_state_locked(self, state: PublicAccessState) -> None:
        self.path.write_text(
            json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
