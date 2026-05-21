from __future__ import annotations

import io
import json
import os
import platform
import queue
import random
import re
import signal
import socket
import subprocess
import tarfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from public_connection_models import (
    ClientCapabilities,
    ClientCommand,
    ClientKind,
    JobCommand,
    JobLogChunkRequest,
    JobRecord,
    JobUpdateRequest,
    PollResponse,
    RegisterClientRequest,
    RegisterClientResponse,
)


_KV_PATTERN = re.compile(r"([a-zA-Z_]+)=([^=]+?)(?=(?:\s+[a-zA-Z_]+=)|$)")
SERVER_RECONNECT_PAUSE_SECONDS = float(
    os.environ.get("PATTERN_CLIENT_RECONNECT_PAUSE_SECONDS", "15")
)
SERVER_TRANSPORT_RETRY_SECONDS = 15.0
SERVER_IO_RETRY_SECONDS = 2.0
SERVER_CONTROL_REQUEST_TIMEOUT_SECONDS = float(
    os.environ.get("PATTERN_CLIENT_CONTROL_TIMEOUT_SECONDS", "90")
)
SERVER_BLOB_REQUEST_TIMEOUT_SECONDS = float(
    os.environ.get("PATTERN_CLIENT_BLOB_TIMEOUT_SECONDS", "3600")
)
SERVER_IDLE_POLL_FLOOR_SECONDS = float(
    os.environ.get("PATTERN_CLIENT_IDLE_POLL_FLOOR_SECONDS", "5")
)
SERVER_IDLE_POLL_CEILING_SECONDS = float(
    os.environ.get("PATTERN_CLIENT_IDLE_POLL_CEILING_SECONDS", "30")
)
SERVER_POLL_JITTER_FRACTION = max(
    0.0,
    min(0.5, float(os.environ.get("PATTERN_CLIENT_POLL_JITTER_FRACTION", "0.15"))),
)
SERVER_REQUEST_RETRY_ATTEMPTS = max(
    1, int(os.environ.get("PATTERN_CLIENT_REQUEST_RETRY_ATTEMPTS", "6"))
)
ARTIFACT_SYNC_INTERVAL_SECONDS = float(
    os.environ.get("PATTERN_CLIENT_ARTIFACT_SYNC_INTERVAL_SECONDS", "20")
)
ARTIFACT_SYNC_STARTUP_DELAY_SECONDS = float(
    os.environ.get("PATTERN_CLIENT_ARTIFACT_SYNC_STARTUP_DELAY_SECONDS", "15")
)
STARTUP_REQUEUE_AVOID_CLIENT_SECONDS = float(
    os.environ.get("PATTERN_CLIENT_STARTUP_REQUEUE_AVOID_CLIENT_SECONDS", "300")
)


class JobLeaseLostError(RuntimeError):
    pass


def _retry_delay_seconds(*, attempt: int, base_seconds: float = 1.0, cap_seconds: float = 15.0) -> float:
    bounded_attempt = max(0, min(int(attempt), 6))
    return min(cap_seconds, base_seconds * (2**bounded_attempt))


def _blob_timeout() -> httpx.Timeout:
    connect_timeout = max(1.0, min(SERVER_BLOB_REQUEST_TIMEOUT_SECONDS, SERVER_CONTROL_REQUEST_TIMEOUT_SECONDS))
    return httpx.Timeout(
        connect=connect_timeout,
        read=None,
        write=None,
        pool=connect_timeout,
    )


def _default_execution_root() -> Path:
    candidate = Path(__file__).resolve().parents[4]
    if (candidate / "src/pattern_learning/dashboard/workflows.py").is_file():
        return candidate
    return Path(__file__).resolve().parents[2]


def _is_local_client(display_name: str, *, hostname: str | None = None) -> bool:
    forced = str(os.getenv("PATTERN_CLIENT_LOCAL", "")).strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True
    display = display_name.strip().lower()
    host = str(hostname or "").strip().lower()
    if host in {"bazzite", "sean-laptop", "sean-desktop"}:
        return True
    return display.startswith(("reward-", "pattern-local-", "local-source-"))


def _preloaded_dataset_roots(execution_root: Path) -> list[str]:
    dataset_root = execution_root / "data" / "datasets"
    if not dataset_root.is_dir():
        return []
    roots: list[str] = []
    for manifest_path in sorted(dataset_root.glob("*/default_simulator_trace_dataset/dataset_manifest.json")):
        roots.append(str(manifest_path.parent.resolve()))
    return roots


def detect_capabilities(
    execution_root: Path,
    *,
    display_name: str,
    client_kind: ClientKind = ClientKind.remote,
    hostname: str | None = None,
    max_concurrency_override: int | None = None,
    instance_key: str | None = None,
) -> ClientCapabilities:
    envs: list[str] = []
    venv_root = execution_root / "venv"
    for env_id in ("phyre", "iphyre", "kinetix", "pooltool", "learning"):
        if (venv_root / env_id / "bin" / "python").is_file():
            envs.append(env_id)
    gpu = (
        subprocess.run(
            ["bash", "-lc", "command -v nvidia-smi >/dev/null 2>&1"],
            check=False,
        ).returncode
        == 0
    )
    local_client = _is_local_client(display_name, hostname=hostname)
    tags: list[str] = []
    if local_client:
        tags.append("local")
    if gpu:
        tags.append("gpu")
    default_max_concurrency = 4 if client_kind == ClientKind.remote else 2 if local_client else 8
    effective_max_concurrency = (
        max(1, int(max_concurrency_override))
        if max_concurrency_override is not None
        else default_max_concurrency
    )
    return ClientCapabilities(
        environments=envs,
        tags=tags,
        gpu=gpu,
        max_concurrency=effective_max_concurrency,
        metadata={
            "execution_root": str(execution_root),
            "client_instance_key": str(instance_key or "").strip(),
            "preloaded_dataset_roots": _preloaded_dataset_roots(execution_root),
        },
    )


@dataclass
class ActiveJob:
    job: JobRecord
    thread: threading.Thread
    process: subprocess.Popen[str] | None = None
    control_command: str | None = None
    control_command_id: str | None = None
    control_command_created_at: datetime | None = None
    control_requested_at: float | None = None
    local_log_path: Path | None = None
    progress_payload: dict[str, object] | None = None
    artifact_signature: tuple[str, ...] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class ClientAgent:
    def __init__(
        self,
        server_url: str,
        *,
        server_api_key: str | None = None,
        client_root: Path,
        execution_root: Path,
        display_name: str,
        client_kind: ClientKind = ClientKind.remote,
        direct_url: str | None = None,
        max_concurrency_override: int | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.server_api_key = str(server_api_key).strip() if server_api_key else ""
        self.client_root = client_root
        self.execution_root = execution_root
        self.display_name = display_name
        self.client_kind = client_kind
        self.direct_url = str(direct_url).strip() if direct_url else ""
        self.max_concurrency_override = (
            max(1, int(max_concurrency_override))
            if max_concurrency_override is not None
            else None
        )
        self.llm_base_url = str(llm_base_url).strip() if llm_base_url else ""
        self.llm_api_key = str(llm_api_key).strip() if llm_api_key else ""
        self.client_id: str | None = None
        self._server_max_concurrency_override: int | None = None
        self._active_jobs: dict[str, ActiveJob] = {}
        self._active_lock = threading.Lock()
        self._registration_lock = threading.Lock()
        self._input_extract_lock = threading.Lock()
        self.instance_key = self._load_or_create_instance_key()
        self._poll_rng = random.Random(self.instance_key)

    def _load_or_create_instance_key(self) -> str:
        state_root = self.client_root / "state"
        state_root.mkdir(parents=True, exist_ok=True)
        key_path = state_root / "client_instance_key.txt"
        try:
            existing = key_path.read_text(encoding="utf-8").strip()
        except OSError:
            existing = ""
        if existing:
            return existing
        instance_key = uuid.uuid4().hex
        key_path.write_text(instance_key + "\n", encoding="utf-8")
        return instance_key

    def _client_state_root(self) -> Path:
        state_root = self.client_root / "state"
        state_root.mkdir(parents=True, exist_ok=True)
        return state_root

    def _client_id_path(self) -> Path:
        return self._client_state_root() / "client_id.txt"

    def _persist_client_id(self, client_id: str) -> None:
        self._client_id_path().write_text(str(client_id).strip() + "\n", encoding="utf-8")

    def _clear_persisted_client_id(self) -> None:
        try:
            self._client_id_path().unlink()
        except FileNotFoundError:
            pass

    def _register_locked(self) -> str:
        request = RegisterClientRequest(
            display_name=self.display_name,
            kind=self.client_kind,
            direct_url=self.direct_url or None,
            hostname=socket.gethostname(),
            platform=platform.platform(),
            capabilities=self._capabilities_for_registration(),
        )
        payload = self._post_json(
            "/api/v1/clients/register",
            request.model_dump(mode="json"),
        )
        response = RegisterClientResponse.model_validate(payload)
        self.client_id = str(response.client["client_id"])
        self._persist_client_id(self.client_id)
        return self.client_id

    def _fresh_registration(self, *, reason: str) -> str:
        previous_client_id = str(self.client_id or "").strip()
        active_jobs = self.current_jobs
        if previous_client_id:
            print(
                f"Re-registering client after {reason}: old_client_id={previous_client_id} active_jobs={len(active_jobs)}",
                flush=True,
            )
        with self._registration_lock:
            current_client_id = str(self.client_id or "").strip()
            if current_client_id and current_client_id != previous_client_id:
                return current_client_id
            if "unknown client" in reason.lower():
                self._server_max_concurrency_override = None
            client_id = self._register_locked()
        self._post_json(
            f"/api/v1/clients/{client_id}/heartbeat",
            {"current_jobs": active_jobs},
        )
        return client_id

    def _discard_client_id(self, *, reason: str) -> None:
        previous_client_id = str(self.client_id or "").strip()
        if not previous_client_id:
            self.client_id = None
            return
        print(
            f"Discarding client registration after {reason}: old_client_id={previous_client_id}",
            flush=True,
        )
        self.client_id = None
        self._clear_persisted_client_id()

    def register(self) -> str:
        with self._registration_lock:
            return self._register_locked()

    def heartbeat(self) -> None:
        if self.client_id is None:
            raise RuntimeError("client not registered")
        client_id = self.client_id
        try:
            self._post_json(
                f"/api/v1/clients/{client_id}/heartbeat",
                {"current_jobs": self.current_jobs},
            )
        except httpx.HTTPStatusError as error:
            if not self._is_unknown_client_error(error):
                raise
            self._fresh_registration(reason=f"heartbeat unknown client {client_id}")

    def _is_job_reportable_to_server(self, active: ActiveJob) -> bool:
        with active.lock:
            control_command = str(active.control_command or "").strip().lower()
            process = active.process
        if control_command in {"pause", "requeue", "cancel"}:
            return False
        # If the child process has already exited but the worker thread is still
        # draining uploads/status updates, stop advertising the job as live.
        if process is not None and process.poll() is not None:
            return False
        return True

    @property
    def current_jobs(self) -> list[str]:
        with self._active_lock:
            return sorted(
                job_id
                for job_id, active in self._active_jobs.items()
                if self._is_job_reportable_to_server(active)
            )

    def poll(self) -> PollResponse:
        if self.client_id is None:
            raise RuntimeError("client not registered")
        client_id = self.client_id
        try:
            payload = self._post_json(f"/api/v1/clients/{client_id}/poll", {})
        except httpx.HTTPStatusError as error:
            if not self._is_unknown_client_error(error):
                raise
            self._fresh_registration(reason=f"poll unknown client {client_id}")
            payload = self._post_json(f"/api/v1/clients/{self.client_id}/poll", {})
        response = PollResponse.model_validate(payload)
        self._apply_server_client_limit(response.client_max_concurrency)
        return response

    def run_forever(
        self,
        *,
        poll_interval_seconds: float = 5.0,
        reconnect_pause_seconds: float = SERVER_RECONNECT_PAUSE_SECONDS,
        transport_retry_seconds: float = SERVER_TRANSPORT_RETRY_SECONDS,
    ) -> None:
        transport_failures = 0
        reconnect_failures = 0
        while True:
            try:
                if self.client_id is None:
                    self.register()
                self.heartbeat()
                response = self.poll()
                if self._handle_client_commands(response.client_commands):
                    return
                self._handle_commands(response.commands)
                for job in response.assignments:
                    self._start_job(job)
                transport_failures = 0
                reconnect_failures = 0
                time.sleep(
                    self._next_poll_delay_seconds(
                        default_seconds=poll_interval_seconds,
                        response=response,
                    )
                )
            except KeyboardInterrupt:
                raise
            except httpx.TransportError as error:
                delay = _retry_delay_seconds(
                    attempt=transport_failures,
                    cap_seconds=max(1.0, transport_retry_seconds),
                )
                transport_failures += 1
                print(
                    f"Server contact failed: {error}. "
                    f"Pausing client control loop and retrying in {delay:.0f}s.",
                    flush=True,
                )
                time.sleep(delay)
            except Exception as error:
                self._discard_client_id(reason=f"control-loop failure: {error}")
                delay = _retry_delay_seconds(
                    attempt=reconnect_failures,
                    cap_seconds=max(1.0, reconnect_pause_seconds),
                )
                reconnect_failures += 1
                print(
                    f"Server contact failed: {error}. "
                    f"Pausing client control loop and retrying in {delay:.0f}s.",
                    flush=True,
                )
                time.sleep(delay)

    def run_heartbeat_forever(
        self,
        *,
        heartbeat_interval_seconds: float = 5.0,
        reconnect_pause_seconds: float = SERVER_RECONNECT_PAUSE_SECONDS,
        transport_retry_seconds: float = SERVER_TRANSPORT_RETRY_SECONDS,
    ) -> None:
        transport_failures = 0
        reconnect_failures = 0
        while True:
            try:
                if self.client_id is None:
                    self.register()
                self.heartbeat()
                transport_failures = 0
                reconnect_failures = 0
                time.sleep(max(1.0, heartbeat_interval_seconds))
            except KeyboardInterrupt:
                raise
            except httpx.TransportError as error:
                delay = _retry_delay_seconds(
                    attempt=transport_failures,
                    cap_seconds=max(1.0, transport_retry_seconds),
                )
                transport_failures += 1
                print(
                    f"Server contact failed: {error}. "
                    f"Pausing direct-client heartbeat and retrying in {delay:.0f}s.",
                    flush=True,
                )
                time.sleep(delay)
            except Exception as error:
                self._discard_client_id(reason=f"heartbeat-loop failure: {error}")
                delay = _retry_delay_seconds(
                    attempt=reconnect_failures,
                    cap_seconds=max(1.0, reconnect_pause_seconds),
                )
                reconnect_failures += 1
                print(
                    f"Server contact failed: {error}. "
                    f"Pausing direct-client heartbeat and retrying in {delay:.0f}s.",
                    flush=True,
                )
                time.sleep(delay)

    def _capabilities_for_registration(self) -> ClientCapabilities:
        hostname = socket.gethostname()
        capabilities = detect_capabilities(
            self.execution_root,
            display_name=self.display_name,
            client_kind=self.client_kind,
            hostname=hostname,
            max_concurrency_override=self.max_concurrency_override,
            instance_key=self.instance_key,
        )
        if self._server_max_concurrency_override is not None:
            capabilities.max_concurrency = self._server_max_concurrency_override
        return capabilities

    def _apply_server_client_limit(self, client_max_concurrency: int | None) -> None:
        if client_max_concurrency is None:
            return
        normalized = max(1, int(client_max_concurrency))
        if normalized == self._server_max_concurrency_override:
            return
        self._server_max_concurrency_override = normalized
        print(
            f"Adopting server-side client max concurrency: {normalized}",
            flush=True,
        )

    def _next_poll_delay_seconds(
        self,
        *,
        default_seconds: float,
        response: PollResponse,
    ) -> float:
        suggested = response.suggested_poll_delay_seconds
        base_delay = float(suggested if suggested is not None else default_seconds)
        if not self.current_jobs and not response.assignments and not response.commands and not response.client_commands:
            base_delay = min(
                SERVER_IDLE_POLL_CEILING_SECONDS,
                max(SERVER_IDLE_POLL_FLOOR_SECONDS, base_delay),
            )
        else:
            base_delay = max(1.0, base_delay)
        jitter_fraction = SERVER_POLL_JITTER_FRACTION
        if jitter_fraction <= 0.0:
            return base_delay
        jitter_span = base_delay * jitter_fraction
        return max(1.0, base_delay + self._poll_rng.uniform(-jitter_span, jitter_span))

    def _start_job(self, job: JobRecord) -> None:
        with self._active_lock:
            existing = self._active_jobs.get(job.job_id)
            if existing is not None:
                if existing.thread.is_alive():
                    return
                self._active_jobs.pop(job.job_id, None)
            thread = threading.Thread(
                target=self._execute_job,
                args=(job,),
                daemon=True,
            )
            self._active_jobs[job.job_id] = ActiveJob(job=job, thread=thread)
            thread.start()

    def _handle_commands(self, commands: list[JobCommand]) -> None:
        for command in commands:
            with self._active_lock:
                active = self._active_jobs.get(command.job_id)
            if active is None:
                self._ack_command(command)
                continue
            with active.lock:
                active.control_command = command.command
                active.control_command_id = command.command_id
                active.control_command_created_at = command.created_at
                active.control_requested_at = time.monotonic()
                process = active.process
            if process is not None and process.poll() is None:
                self._terminate_process_group(process, grace_seconds=5.0)

    def _handle_client_commands(self, commands: list[ClientCommand]) -> bool:
        shutdown_requested = False
        for command in commands:
            if str(command.command).strip().lower() == "shutdown":
                self._ack_client_command(command)
                shutdown_requested = True
                continue
            self._ack_client_command(command)
        if not shutdown_requested:
            return False
        print("Shutdown requested by server; stopping client.", flush=True)
        with self._active_lock:
            active_jobs = list(self._active_jobs.values())
        for active in active_jobs:
            with active.lock:
                process = active.process
            if process is not None and process.poll() is None:
                self._terminate_process_group(process, grace_seconds=5.0)
        return True

    def _terminate_process_group(
        self,
        process: subprocess.Popen[str],
        *,
        grace_seconds: float = 5.0,
    ) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=max(0.1, grace_seconds))
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

    def _execute_job(self, job: JobRecord) -> None:
        job_id = job.job_id
        self._retry_update_job(
            job_id,
            JobUpdateRequest(client_id=self.client_id, status="starting"),
        )
        state_root = self.client_root / "state"
        state_root.mkdir(parents=True, exist_ok=True)
        local_log_path = state_root / f"{job_id}.log"
        local_log_path.write_text("", encoding="utf-8")
        process: subprocess.Popen[str] | None = None
        try:
            prepared_job = self._prepare_job_inputs(job)
            prepared_job = self._apply_llm_runtime_overrides(prepared_job)
            dependency_pause_reason = self._prerequisite_pause_reason(prepared_job)
            if dependency_pause_reason:
                self._retry_update_job(
                    job_id,
                    JobUpdateRequest(
                        client_id=self.client_id,
                        status="paused",
                        message=f"Waiting for prerequisite: {dependency_pause_reason}",
                        metadata_patch={
                            "progress": {
                                "phase": "Waiting for prerequisite",
                                "summary": dependency_pause_reason,
                            }
                        },
                    ),
                )
                return
            argv = [str(part) for part in prepared_job.spec.argv]
            if prepared_job.spec.cwd:
                raw_cwd = Path(prepared_job.spec.cwd)
                cwd = raw_cwd if raw_cwd.is_absolute() else (self.execution_root / raw_cwd)
            else:
                cwd = self.execution_root
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in prepared_job.spec.env.items()})
            if self.llm_base_url:
                env["LLM_ROUTER_BASE_URL"] = self.llm_base_url
                env["OPENAI_BASE_URL"] = self.llm_base_url
            if self.llm_api_key:
                env["LLM_ROUTER_API_KEY"] = self.llm_api_key
                env["OPENAI_API_KEY"] = self.llm_api_key
            project_src = str(self.execution_root / "src")
            existing_pythonpath = env.get("PYTHONPATH", "").strip()
            env["PYTHONPATH"] = (
                project_src
                if not existing_pythonpath
                else os.pathsep.join([project_src, existing_pythonpath])
            )
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("MPLBACKEND", "Agg")

            process = subprocess.Popen(
                argv,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with self._active_lock:
                active = self._active_jobs[job_id]
            with active.lock:
                active.process = process
                active.local_log_path = local_log_path
            self._retry_update_job(
                job_id,
                JobUpdateRequest(client_id=self.client_id, status="running"),
            )
            job_started_monotonic = time.monotonic()

            output_queue: queue.Queue[str | None] = queue.Queue()
            reader = threading.Thread(
                target=self._read_process_output,
                args=(process, output_queue, local_log_path),
                daemon=True,
            )
            reader.start()

            pending_lines: list[str] = []
            last_flush = time.monotonic()
            last_heartbeat = time.monotonic()
            last_artifact_sync = 0.0
            last_progress_payload: dict[str, object] | None = None
            heartbeat_interval = 5.0
            while True:
                try:
                    line = output_queue.get(timeout=1.0)
                except queue.Empty:
                    line = None
                if line is not None:
                    pending_lines.append(line)
                    progress_payload = self._progress_payload_from_line(line)
                    if progress_payload is not None and progress_payload != last_progress_payload:
                        if self._try_update_job(
                            job_id,
                            JobUpdateRequest(
                                client_id=self.client_id,
                                status="running",
                                metadata_patch={"progress": progress_payload},
                            ),
                        ):
                            last_progress_payload = progress_payload
                            with active.lock:
                                active.progress_payload = progress_payload
                if pending_lines and (
                    len(pending_lines) >= 50 or time.monotonic() - last_flush >= 2.0
                ):
                    if self._try_append_logs(job_id, pending_lines):
                        pending_lines = []
                        last_flush = time.monotonic()
                if (
                    time.monotonic() - job_started_monotonic >= ARTIFACT_SYNC_STARTUP_DELAY_SECONDS
                    and time.monotonic() - last_artifact_sync >= ARTIFACT_SYNC_INTERVAL_SECONDS
                ):
                    if self._try_sync_artifacts(
                        prepared_job,
                        local_log_path=local_log_path,
                        active=active,
                    ):
                        last_artifact_sync = time.monotonic()
                if time.monotonic() - last_heartbeat >= heartbeat_interval:
                    if self._try_heartbeat():
                        last_heartbeat = time.monotonic()
                with active.lock:
                    control_command = active.control_command
                    control_requested_at = active.control_requested_at
                if (
                    control_command
                    and control_requested_at is not None
                    and process.poll() is None
                    and time.monotonic() - control_requested_at >= 5.0
                ):
                    self._terminate_process_group(process, grace_seconds=0.1)
                code = process.poll()
                if code is not None and output_queue.empty():
                    if pending_lines:
                        self._retry_append_logs(job_id, pending_lines)
                    self._finalize_job(prepared_job, exit_code=int(code), local_log_path=local_log_path)
                    return
        except JobLeaseLostError as error:
            print(f"Stopping stale worker for {job_id}: {error}", flush=True)
            if process is not None and process.poll() is None:
                self._terminate_process_group(process, grace_seconds=5.0)
        except Exception as error:
            summary = f"{type(error).__name__}: {error}"
            traceback_text = traceback.format_exc().rstrip()
            with local_log_path.open("a", encoding="utf-8") as handle:
                handle.write(traceback_text + "\n")
            print(f"Job execution failed for {job_id}: {summary}", flush=True)
            if process is not None and process.poll() is None:
                self._terminate_process_group(process, grace_seconds=5.0)
            try:
                self._retry_append_logs(job_id, traceback_text.splitlines()[-50:])
            except Exception:
                pass
            try:
                metadata_patch: dict[str, object] = {
                    "progress": {
                        "phase": "Queued for retry",
                        "summary": summary[:220],
                    }
                }
                if self._is_retryable_server_error(error) and self.client_id:
                    metadata_patch["avoid_client_id"] = self.client_id
                    metadata_patch["avoid_client_until"] = (
                        datetime.now(timezone.utc)
                        + timedelta(seconds=max(0.0, STARTUP_REQUEUE_AVOID_CLIENT_SECONDS))
                    ).isoformat()
                self._retry_update_job(
                    job_id,
                    JobUpdateRequest(
                        client_id=self.client_id,
                        status="queued",
                        message=f"Client startup failed; re-queued automatically: {summary[:180]}",
                        metadata_patch=metadata_patch,
                    ),
                )
            except Exception:
                pass
        finally:
            with self._active_lock:
                self._active_jobs.pop(job_id, None)

    def _finalize_job(self, job: JobRecord, *, exit_code: int, local_log_path: Path) -> None:
        with self._active_lock:
            active = self._active_jobs.get(job.job_id)
        command = None
        command_id = None
        command_created_at = None
        if active is not None:
            with active.lock:
                command = active.control_command
                command_id = active.control_command_id
                command_created_at = active.control_command_created_at
        artifact_paths = self._artifact_paths_for_job(job, local_log_path=local_log_path)
        extracted_paths: list[str] = []
        if artifact_paths:
            try:
                extracted_paths = self._upload_artifact_bundle(job.job_id, artifact_paths)
            except JobLeaseLostError:
                raise
            except Exception as error:
                print(f"Final artifact upload failed for {job.job_id}: {error}", flush=True)
                extracted_paths = []
        if command == "pause":
            status = "paused"
        elif command == "cancel":
            status = "cancelled"
        elif command == "requeue":
            status = "queued"
        else:
            status = "succeeded" if exit_code == 0 else "failed"
        payload = JobUpdateRequest(
            client_id=self.client_id,
            status=status,
            exit_code=exit_code if status in {"succeeded", "failed"} else None,
            metadata_patch={
                "uploaded_artifact_count": len(extracted_paths),
                "uploaded_artifacts_preview": extracted_paths[:8],
                "progress": self._final_progress_payload(
                    status=status,
                    active=active,
                    exit_code=exit_code,
                ),
            },
        )
        self._retry_update_job(job.job_id, payload)
        if command and command_id and command_created_at is not None:
            self._retry_ack_command(
                JobCommand(
                    command_id=command_id,
                    job_id=job.job_id,
                    command=command,
                    created_at=command_created_at,
                )
            )

    def _read_process_output(
        self,
        process: subprocess.Popen[str],
        output_queue: queue.Queue[str | None],
        local_log_path: Path,
    ) -> None:
        assert process.stdout is not None
        with local_log_path.open("a", encoding="utf-8") as handle:
            for raw_line in process.stdout:
                line = raw_line.rstrip("\n")
                handle.write(line + "\n")
                handle.flush()
                output_queue.put(line)

    def _artifact_paths_for_job(
        self,
        job: JobRecord,
        *,
        local_log_path: Path,
    ) -> list[Path]:
        metadata = dict(job.spec.metadata or {})
        candidates: list[Path] = [local_log_path]
        explicit = metadata.get("artifact_paths")
        if isinstance(explicit, list):
            for item in explicit:
                if isinstance(item, str) and item.strip():
                    candidates.append(Path(item))
        for key in (
            "run_dir",
            "evaluation_run_dir",
            "log_path",
            "stdout_path",
            "stderr_path",
            "snapshot_path",
        ):
            raw = metadata.get(key)
            if isinstance(raw, str) and raw.strip():
                candidates.append(Path(raw))
        output_path = metadata.get("output_path")
        if isinstance(output_path, str) and output_path.strip():
            output_candidate = Path(output_path)
            candidates.append(output_candidate)
            if output_candidate.name == "pattern_library.json":
                candidates.append(output_candidate.parent)
        request_payload = metadata.get("request")
        if isinstance(request_payload, dict):
            for key in ("output_dir", "report_root", "pattern_run_dir", "source_supervised_run_dir"):
                raw = request_payload.get(key)
                if isinstance(raw, str) and raw.strip():
                    candidates.append(Path(raw))
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            resolved = candidate if candidate.is_absolute() else (self.execution_root / candidate)
            try:
                normalized = str(resolved.resolve())
            except OSError:
                continue
            if normalized in seen or not resolved.exists():
                continue
            seen.add(normalized)
            unique.append(resolved)
        return unique

    def _try_sync_artifacts(
        self,
        job: JobRecord,
        *,
        local_log_path: Path,
        active: ActiveJob | None = None,
    ) -> bool:
        artifact_paths = self._artifact_paths_for_job(job, local_log_path=local_log_path)
        if not artifact_paths:
            return True
        artifact_signature = self._artifact_signature(artifact_paths)
        if active is not None:
            with active.lock:
                if active.artifact_signature == artifact_signature:
                    return True
        try:
            self._upload_artifact_bundle(job.job_id, artifact_paths)
            if active is not None:
                with active.lock:
                    active.artifact_signature = artifact_signature
            return True
        except JobLeaseLostError:
            raise
        except httpx.HTTPStatusError as error:
            if self._is_unknown_client_error(error):
                self._fresh_registration(reason=f"artifact upload unknown client {self.client_id or 'missing'}")
                self._upload_artifact_bundle(job.job_id, artifact_paths)
                return True
            print(f"Artifact sync failed for {job.job_id}: {error}", flush=True)
            return False
        except Exception as error:
            print(f"Artifact sync failed for {job.job_id}: {error}", flush=True)
            return False

    def _artifact_signature(self, paths: list[Path]) -> tuple[str, ...]:
        execution_root = self.execution_root.resolve()
        signature: list[str] = []
        seen: set[str] = set()
        for path in sorted(paths, key=lambda item: str(item)):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if not resolved.exists():
                continue
            descendants = [resolved]
            if resolved.is_dir():
                descendants.extend(sorted(resolved.rglob("*")))
            for descendant in descendants:
                try:
                    stat_result = descendant.stat()
                except OSError:
                    continue
                try:
                    relpath = descendant.relative_to(execution_root).as_posix()
                except ValueError:
                    relpath = str(descendant)
                if relpath in seen:
                    continue
                seen.add(relpath)
                kind = "d" if descendant.is_dir() else "f"
                signature.append(
                    f"{kind}:{relpath}:{int(stat_result.st_size)}:{int(stat_result.st_mtime_ns)}"
                )
        return tuple(signature)

    def _prepare_job_inputs(self, job: JobRecord) -> JobRecord:
        plan = self._get_json(f"/api/v1/jobs/{job.job_id}/inputs")
        bundle_items = plan.get("bundle_items") if isinstance(plan, dict) else []
        rewrite_prefixes = plan.get("rewrite_prefixes") if isinstance(plan, dict) else []
        if not isinstance(bundle_items, list):
            bundle_items = []
        if not isinstance(rewrite_prefixes, list):
            rewrite_prefixes = []
        if any(self._bundle_item_missing(item) for item in bundle_items):
            bundle = self._get_bytes(f"/api/v1/jobs/{job.job_id}/input-bundle")
            self._extract_input_bundle(bundle)
        self._rewrite_downloaded_inputs(bundle_items, rewrite_prefixes)
        return self._rewrite_job_paths(job, rewrite_prefixes)

    def _bundle_item_missing(self, item: object) -> bool:
        if not isinstance(item, dict):
            return False
        target_kind = item.get("target_kind")
        if isinstance(target_kind, str) and target_kind == "run-dir":
            # Resumed jobs depend on the server's latest run-dir state, so an
            # already-present local directory is not enough to skip refresh.
            return True
        target_relpath = item.get("target_relpath")
        if not isinstance(target_relpath, str) or not target_relpath.strip():
            return False
        return not (self.execution_root / target_relpath).exists()

    def _extract_input_bundle(self, bundle: bytes) -> None:
        execution_root = self.execution_root.resolve()
        with self._input_extract_lock:
            with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
                for member in archive.getmembers():
                    member_path = Path(member.name)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        continue
                    destination = (execution_root / member_path).resolve()
                    try:
                        destination.relative_to(execution_root)
                    except ValueError:
                        continue
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        continue
                    source_handle = archive.extractfile(member)
                    if source_handle is None:
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temp_path = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
                    try:
                        with temp_path.open("wb") as handle:
                            while True:
                                chunk = source_handle.read(1024 * 1024)
                                if not chunk:
                                    break
                                handle.write(chunk)
                        os.replace(temp_path, destination)
                    finally:
                        try:
                            source_handle.close()
                        except OSError:
                            pass
                        try:
                            if temp_path.exists():
                                temp_path.unlink()
                        except OSError:
                            pass

    def _rewrite_downloaded_inputs(
        self,
        bundle_items: list[object],
        rewrite_prefixes: list[object],
    ) -> None:
        replacements = self._local_rewrite_pairs(rewrite_prefixes)
        if not replacements:
            return
        seen_files: set[str] = set()
        for item in bundle_items:
            if not isinstance(item, dict):
                continue
            target_relpath = item.get("target_relpath")
            if not isinstance(target_relpath, str) or not target_relpath.strip():
                continue
            target_path = (self.execution_root / target_relpath).resolve()
            candidates: list[Path]
            if target_path.is_dir():
                candidates = [
                    path
                    for path in target_path.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml", ".txt"}
                ]
            elif target_path.is_file():
                candidates = [target_path]
            else:
                continue
            for candidate in candidates:
                key = str(candidate)
                if key in seen_files:
                    continue
                seen_files.add(key)
                self._rewrite_text_file(candidate, replacements)

    def _rewrite_text_file(
        self,
        path: Path,
        replacements: list[tuple[str, str]],
    ) -> None:
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        updated = original
        tokens: list[tuple[str, str]] = []
        for index, (source_prefix, target_prefix) in enumerate(replacements):
            if source_prefix not in updated:
                continue
            token = f"__PATTERN_CLIENT_REWRITE_{index}__"
            updated = updated.replace(source_prefix, token)
            tokens.append((token, target_prefix))
        for token, target_prefix in tokens:
            updated = updated.replace(token, target_prefix)
        if updated != original:
            path.write_text(updated, encoding="utf-8")

    def _rewrite_job_paths(self, job: JobRecord, rewrite_prefixes: list[object]) -> JobRecord:
        replacements = self._local_rewrite_pairs(rewrite_prefixes)
        execution_root_prefix = str(self.execution_root.resolve()).rstrip("/")

        def rewrite_value(value: object) -> object:
            if isinstance(value, str):
                updated = value
                if execution_root_prefix and updated.startswith(execution_root_prefix):
                    return updated
                for source_prefix, target_prefix in replacements:
                    if updated.startswith(source_prefix):
                        updated = target_prefix + updated[len(source_prefix):]
                        break
                return updated
            if isinstance(value, list):
                return [rewrite_value(item) for item in value]
            if isinstance(value, dict):
                return {str(key): rewrite_value(item) for key, item in value.items()}
            return value

        payload = json.loads(job.model_dump_json())
        payload["spec"] = rewrite_value(payload["spec"])
        return JobRecord.model_validate(payload)

    def _apply_llm_runtime_overrides(self, job: JobRecord) -> JobRecord:
        if not self.llm_base_url and not self.llm_api_key:
            return job
        payload = json.loads(job.model_dump_json())
        spec = payload.get("spec", {})
        argv = spec.get("argv")
        if self.llm_base_url and isinstance(argv, list):
            spec["argv"] = self._rewrite_argv_base_urls(argv)
        env = spec.get("env")
        if not isinstance(env, dict):
            env = {}
        if self.llm_base_url:
            env["LLM_ROUTER_BASE_URL"] = self.llm_base_url
            env["OPENAI_BASE_URL"] = self.llm_base_url
        if self.llm_api_key:
            env["LLM_ROUTER_API_KEY"] = self.llm_api_key
            env["OPENAI_API_KEY"] = self.llm_api_key
        spec["env"] = env
        payload["spec"] = spec
        return JobRecord.model_validate(payload)

    def _prerequisite_pause_reason(self, job: JobRecord) -> str | None:
        action = str(job.spec.action or "").strip()
        metadata = dict(job.spec.metadata or {})
        if action == "evaluate-deepphy":
            pattern_run_dir = str(metadata.get("pattern_run_dir") or "").strip()
            if not pattern_run_dir:
                request = metadata.get("request")
                if isinstance(request, dict):
                    pattern_run_dir = str(request.get("pattern_run_dir") or "").strip()
            if not pattern_run_dir:
                return None
            library_path = Path(pattern_run_dir) / "pattern_library.json"
            if not library_path.is_file():
                return f"{library_path} not found"
            argv = [str(part) for part in job.spec.argv]
            if len(argv) >= 4 and argv[1] == "-c" and "Waiting for" in argv[2]:
                wrapped_target = str(argv[3]).strip()
                if wrapped_target:
                    wrapped_target_path = Path(wrapped_target)
                    if not wrapped_target_path.is_file():
                        return f"{wrapped_target_path} not found"
        if action == "analyze-supervised-consistency":
            request = metadata.get("request")
            request_dict = request if isinstance(request, dict) else {}
            trace_path = str(metadata.get("trace_path") or request_dict.get("trace_path") or "").strip()
            if trace_path:
                resolved_trace = Path(trace_path)
                if not resolved_trace.is_absolute():
                    resolved_trace = self.execution_root / resolved_trace
                if not resolved_trace.is_file():
                    return f"{resolved_trace} not found"
        return None

    def _rewrite_argv_base_urls(self, argv: list[object]) -> list[str]:
        if not self.llm_base_url:
            return [str(part) for part in argv]
        rewritten = [str(part) for part in argv]
        flags = {"--ollama-base-url", "--generator-base-url", "--prediction-base-url"}
        for index, value in enumerate(rewritten[:-1]):
            if value in flags:
                rewritten[index + 1] = self.llm_base_url
        return rewritten

    def _local_rewrite_pairs(self, rewrite_prefixes: list[object]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for item in rewrite_prefixes:
            if not isinstance(item, dict):
                continue
            source_prefix = item.get("source_prefix")
            target_relpath = item.get("target_relpath")
            if not isinstance(source_prefix, str) or not source_prefix.strip():
                continue
            if not isinstance(target_relpath, str):
                continue
            target_prefix = str((self.execution_root / target_relpath).resolve())
            pairs.append((source_prefix.rstrip("/"), target_prefix.rstrip("/")))
        # Older synced run artifacts may still embed legacy runtime paths from the
        # original worker filesystem. Translate those into the local synced bundle too.
        pairs.extend(
            [
                (
                    "/var/home/sean/Services/Work/pattern-learning-client",
                    str(self.execution_root.resolve()),
                ),
                ("/var/home/sean/Services/venv", str((self.execution_root / "venv").resolve())),
                ("/var/home/sean/Services/configs", str((self.execution_root / "configs").resolve())),
                ("/var/home/sean/Services/data", str((self.execution_root / "data").resolve())),
                ("/var/home/sean/Services", str(self.execution_root.resolve())),
                ("/opt/pattern-learning-stack/data", str((self.execution_root / "data").resolve())),
                ("/opt/pattern-learning-stack/configs", str((self.execution_root / "configs").resolve())),
                ("/opt/pattern-learning-stack", str(self.execution_root.resolve())),
            ]
        )
        pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
        return pairs

    def _upload_artifact_bundle(self, job_id: str, paths: list[Path]) -> list[str]:
        bundle = io.BytesIO()
        added: set[str] = set()
        with tarfile.open(fileobj=bundle, mode="w:gz") as tar:
            for path in paths:
                try:
                    relpath = path.resolve().relative_to(self.execution_root.resolve())
                except ValueError:
                    if path.name.startswith(job_id):
                        relpath = Path("v2/client/state") / path.name
                    else:
                        continue
                arcname = relpath.as_posix()
                if arcname in added:
                    continue
                tar.add(path, arcname=arcname, recursive=True)
                added.add(arcname)
        response = self._post_bytes(
            f"/api/v1/jobs/{job_id}/artifact-bundle",
            bundle.getvalue(),
            headers={
                "content-type": "application/gzip",
                "x-client-id": str(self.client_id or ""),
            },
        )
        return list(response.get("extracted_paths", []))

    def _ack_command(self, command: JobCommand) -> None:
        self._post_json(
            f"/api/v1/jobs/{command.job_id}/commands/{command.command_id}/ack",
            {},
        )

    def _retry_ack_command(self, command: JobCommand) -> None:
        while True:
            try:
                self._ack_command(command)
                return
            except Exception as error:
                print(f"Command ack failed for {command.job_id}: {error}", flush=True)
                time.sleep(SERVER_IO_RETRY_SECONDS)

    def _ack_client_command(self, command: ClientCommand) -> None:
        if self.client_id is None:
            raise RuntimeError("client not registered")
        self._post_json(
            f"/api/v1/clients/{self.client_id}/commands/{command.command_id}/ack",
            {},
        )

    def _append_logs(self, job_id: str, lines: list[str]) -> None:
        self._post_json(
            f"/api/v1/jobs/{job_id}/logs",
            JobLogChunkRequest(client_id=self.client_id, lines=lines).model_dump(mode="json"),
        )

    def _update_job(self, job_id: str, request: JobUpdateRequest) -> None:
        self._put_json(
            f"/api/v1/jobs/{job_id}/status",
            request.model_dump(mode="json"),
        )

    def _try_heartbeat(self) -> bool:
        try:
            self.heartbeat()
            return True
        except Exception as error:
            print(f"Heartbeat failed during job execution: {error}", flush=True)
            return False

    def _try_append_logs(self, job_id: str, lines: list[str]) -> bool:
        try:
            self._append_logs(job_id, lines)
            return True
        except JobLeaseLostError:
            raise
        except httpx.HTTPStatusError as error:
            if self._is_unknown_client_error(error):
                self._fresh_registration(reason=f"log upload unknown client {self.client_id or 'missing'}")
                self._append_logs(job_id, lines)
                return True
            print(f"Log flush failed for {job_id}: {error}", flush=True)
            return False
        except Exception as error:
            print(f"Log flush failed for {job_id}: {error}", flush=True)
            return False

    def _retry_append_logs(self, job_id: str, lines: list[str]) -> None:
        while True:
            if self._try_append_logs(job_id, lines):
                return
            time.sleep(SERVER_IO_RETRY_SECONDS)

    def _try_update_job(self, job_id: str, request: JobUpdateRequest) -> bool:
        try:
            self._update_job(job_id, request)
            return True
        except JobLeaseLostError:
            current_client_id = str(self.client_id or "").strip()
            request_client_id = str(request.client_id or "").strip()
            if current_client_id and current_client_id != request_client_id:
                self._update_job(
                    job_id,
                    request.model_copy(update={"client_id": current_client_id}),
                )
                return True
            raise
        except httpx.HTTPStatusError as error:
            if self._is_unknown_client_error(error):
                self._fresh_registration(reason=f"job update unknown client {self.client_id or 'missing'}")
                self._update_job(
                    job_id,
                    request.model_copy(update={"client_id": self.client_id}),
                )
                return True
            print(f"Job update failed for {job_id}: {error}", flush=True)
            return False
        except Exception as error:
            print(f"Job update failed for {job_id}: {error}", flush=True)
            return False

    def _retry_update_job(self, job_id: str, request: JobUpdateRequest) -> None:
        while True:
            if self._try_update_job(job_id, request):
                return
            time.sleep(SERVER_IO_RETRY_SECONDS)

    def _progress_payload_from_line(self, line: str) -> dict[str, object] | None:
        text = str(line).strip()
        if not text:
            return None
        if ": " in text and text.startswith("["):
            event_prefix, _, summary = text.partition(": ")
            event = event_prefix.split("] ", 1)[-1].strip()
            fields = self._parse_summary_fields(summary)
        else:
            event = ""
            summary = text
            fields = {}

        if event == "cli.pattern_learn.samples_loaded":
            return {
                "phase": "Samples loaded",
                "summary": f"Loaded {fields.get('sample_count', '?')} train and {fields.get('validation_sample_count', '?')} validation samples",
                "current": 0,
                "total": int(fields.get("primitive_label_count", 0) or 0) or None,
                "fraction": 0.0,
            }
        if event == "patterns.learning.label.start":
            current = self._to_int(fields.get("index"))
            total = self._to_int(fields.get("total_labels"))
            label = str(fields.get("label") or "").strip()
            fraction = ((current - 1) / total) if current is not None and total else 0.0
            summary = label or f"Label {current or '?'}"
            return {
                "phase": "Learning labels",
                "summary": summary,
                "current": current,
                "total": total,
                "fraction": fraction,
            }
        if event == "patterns.learning.label.done":
            current = self._to_int(fields.get("index"))
            total = self._to_int(fields.get("total_labels"))
            label = str(fields.get("label") or "").strip()
            fraction = (current / total) if current is not None and total else None
            return {
                "phase": "Learning labels",
                "summary": f"Completed {label}" if label else "Completed label",
                "current": current,
                "total": total,
                "fraction": fraction,
            }
        if event == "dashboard.workflows.library_eval.case_start":
            case_id = str(fields.get("case_id") or "").strip()
            return {
                "phase": "Library holdout evaluation",
                "summary": case_id or "Starting evaluation case",
            }
        if event == "dashboard.workflows.library_eval.done":
            return {
                "phase": "Library holdout evaluation",
                "summary": "Completed",
                "fraction": 1.0,
            }
        if event == "patterns.analysis.consistency.start":
            runs = self._to_int(fields.get("runs"))
            windows = (
                self._to_int(fields.get("windows"))
                or self._to_int(fields.get("window_count"))
                or self._to_int(fields.get("checkpoints"))
            )
            repeats = self._to_int(fields.get("sampling_repeats"))
            summary_bits: list[str] = []
            if runs is not None:
                summary_bits.append(f"{runs} runs")
            if windows is not None:
                summary_bits.append(f"{windows} windows")
            if repeats is not None:
                summary_bits.append(f"{repeats} repeats")
            return {
                "phase": "Consistency analysis",
                "summary": " · ".join(summary_bits) if summary_bits else "Preparing consistency analysis",
                "fraction": 0.0,
            }
        if event == "patterns.analysis.consistency.repeat.start":
            current = self._to_int(fields.get("repeat"))
            total = self._to_int(fields.get("total_repeats"))
            fraction = ((current - 1) / total) if current is not None and total else 0.0
            return {
                "phase": "Consistency analysis",
                "summary": "Sampling repeats",
                "current": current,
                "total": total,
                "fraction": fraction,
            }
        if event == "patterns.analysis.consistency.repeat.done":
            current = self._to_int(fields.get("repeat"))
            total = self._to_int(fields.get("total_repeats"))
            fraction = (current / total) if current is not None and total else None
            return {
                "phase": "Consistency analysis",
                "summary": "Sampling repeats",
                "current": current,
                "total": total,
                "fraction": fraction,
            }
        if event == "patterns.analysis.consistency.checkpoint.start":
            current = self._to_int(fields.get("checkpoint"))
            total = self._to_int(fields.get("total_checkpoints"))
            budget = self._to_int(fields.get("budget"))
            fraction = ((current - 1) / total) if current is not None and total else 0.0
            return {
                "phase": "Consistency analysis",
                "summary": f"Budget {budget or '?'}",
                "current": current,
                "total": total,
                "fraction": fraction,
            }
        if event == "patterns.analysis.consistency.checkpoint.done":
            current = self._to_int(fields.get("checkpoint"))
            total = self._to_int(fields.get("total_checkpoints"))
            budget = self._to_int(fields.get("budget"))
            similarity = str(fields.get("mean_similarity") or "").strip()
            fraction = (current / total) if current is not None and total else None
            summary_bits = [f"Budget {budget}" if budget is not None else ""]
            if similarity:
                summary_bits.append(f"similarity {similarity}")
            summary = " · ".join(bit for bit in summary_bits if bit)
            return {
                "phase": "Consistency analysis",
                "summary": summary or "Checkpoint complete",
                "current": current,
                "total": total,
                "fraction": fraction,
            }
        if event == "patterns.analysis.consistency.completed":
            runs = self._to_int(fields.get("runs"))
            windows = self._to_int(fields.get("windows"))
            summary = "Completed"
            if runs is not None and windows is not None:
                summary = f"Completed {runs} runs across {windows} windows"
            return {
                "phase": "Consistency analysis",
                "summary": summary,
                "fraction": 1.0,
            }
        if event == "cli.pattern_learn.completed":
            return {
                "phase": "Learning complete",
                "summary": f"Average score {fields.get('average_score', '?')}",
                "fraction": 1.0,
            }
        if event in {"dashboard.workflows.supervised.start", "dashboard.workflows.child.start"}:
            return {
                "phase": "Starting",
                "summary": summary[:220],
            }
        if event and event.startswith("patterns.learning.lm_request."):
            return {
                "phase": "LLM generation",
                "summary": summary[:220],
            }
        if event and event.startswith("dashboard.workflows."):
            return {
                "phase": "Workflow",
                "summary": summary[:220] if summary else event,
            }
        if event and event.startswith("cli.evaluate_deepphy"):
            return {
                "phase": "DeepPHY evaluation",
                "summary": summary[:220] if summary else event,
            }
        if text.startswith("env_id\tbenchmark\ttask_count\tsuccess_count\tsuccess_rate"):
            return {
                "phase": "DeepPHY evaluation",
                "summary": "Writing benchmark summary",
                "fraction": 1.0,
            }
        return None

    def _parse_summary_fields(self, summary: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for key, value in _KV_PATTERN.findall(summary):
            parsed[key.strip()] = value.strip().rstrip(",")
        return parsed

    def _to_int(self, value: object) -> int | None:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    def _final_progress_payload(
        self,
        *,
        status: str,
        active: ActiveJob | None,
        exit_code: int,
    ) -> dict[str, object]:
        if status == "succeeded":
            return {"phase": "Completed", "summary": "Done", "fraction": 1.0}
        if status == "failed":
            return {
                "phase": "Failed",
                "summary": f"Exit code {exit_code}",
                "fraction": 1.0,
            }
        if status == "paused":
            return {"phase": "Paused", "summary": "Paused by command"}
        if status == "cancelled":
            return {"phase": "Cancelled", "summary": "Cancelled by command"}
        if status == "queued":
            return {"phase": "Queued", "summary": "Queued to resume"}
        if active is not None and active.progress_payload is not None:
            return active.progress_payload
        return {"phase": status.title(), "summary": status.title()}

    def _post_json(self, path: str, payload: dict) -> dict:
        return self._request_json(
            "POST",
            path,
            json_payload=payload,
            timeout=SERVER_CONTROL_REQUEST_TIMEOUT_SECONDS,
        )

    def _is_unknown_client_error(self, error: httpx.HTTPStatusError) -> bool:
        if error.response.status_code != 404:
            return False
        try:
            detail = error.response.json().get("detail")
        except ValueError:
            detail = error.response.text
        return "unknown client" in str(detail or "").strip().lower()

    def _is_retryable_server_error(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {408, 425, 429, 500, 502, 503, 504}
        return False

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict | None = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | httpx.Timeout,
    ) -> dict:
        attempt = 0
        while True:
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.request(
                        method,
                        f"{self.server_url}{path}",
                        json=json_payload,
                        content=content,
                        headers=self._server_request_headers(headers),
                    )
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as error:
                        if error.response.status_code == 409:
                            raise JobLeaseLostError(error.response.text) from error
                        raise
                    return response.json()
            except Exception as error:
                if self._is_retryable_server_error(error) and attempt < SERVER_REQUEST_RETRY_ATTEMPTS:
                    delay = _retry_delay_seconds(
                        attempt=attempt,
                        cap_seconds=max(1.0, SERVER_TRANSPORT_RETRY_SECONDS),
                    )
                    attempt += 1
                    print(
                        f"Retryable server error for {method} {path}: {error}. "
                        f"Retrying in {delay:.1f}s.",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                raise

    def _request_bytes(self, path: str, *, timeout: float) -> bytes:
        attempt = 0
        while True:
            try:
                with httpx.Client(timeout=_blob_timeout()) as client:
                    headers = self._server_request_headers()
                    stream_context = (
                        client.stream(
                            "GET",
                            f"{self.server_url}{path}",
                            headers=headers,
                        )
                        if headers is not None
                        else client.stream("GET", f"{self.server_url}{path}")
                    )
                    with stream_context as response:
                        response.raise_for_status()
                        chunks = bytearray()
                        for chunk in response.iter_bytes():
                            chunks.extend(chunk)
                    return bytes(chunks)
            except Exception as error:
                if self._is_retryable_server_error(error) and attempt < SERVER_REQUEST_RETRY_ATTEMPTS:
                    delay = _retry_delay_seconds(
                        attempt=attempt,
                        cap_seconds=max(1.0, SERVER_TRANSPORT_RETRY_SECONDS),
                    )
                    attempt += 1
                    print(
                        f"Retryable server error for GET {path}: {error}. "
                        f"Retrying in {delay:.1f}s.",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                raise

    def _put_json(self, path: str, payload: dict) -> dict:
        return self._request_json(
            "PUT",
            path,
            json_payload=payload,
            timeout=SERVER_CONTROL_REQUEST_TIMEOUT_SECONDS,
        )

    def _get_json(self, path: str) -> dict:
        return self._request_json("GET", path, timeout=SERVER_CONTROL_REQUEST_TIMEOUT_SECONDS)

    def _get_bytes(self, path: str) -> bytes:
        return self._request_bytes(path, timeout=SERVER_BLOB_REQUEST_TIMEOUT_SECONDS)

    def _post_bytes(self, path: str, content: bytes, *, headers: dict[str, str]) -> dict:
        return self._request_json(
            "POST",
            path,
            content=content,
            headers=headers,
            timeout=_blob_timeout(),
        )

    def _server_request_headers(self, headers: dict[str, str] | None = None) -> dict[str, str] | None:
        merged = dict(headers or {})
        if self.server_api_key:
            merged.setdefault("Authorization", f"Bearer {self.server_api_key}")
            merged.setdefault("X-API-Key", self.server_api_key)
        return merged or None
