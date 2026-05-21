from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ClientKind(str, Enum):
    remote = "remote"
    direct = "direct"


class ClientCapabilities(BaseModel):
    environments: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    gpu: bool = False
    max_concurrency: int = 4
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegisterClientRequest(BaseModel):
    display_name: str
    kind: ClientKind = ClientKind.remote
    direct_url: str | None = None
    version: str = "0.1.0"
    hostname: str = ""
    platform: str = ""
    capabilities: ClientCapabilities = Field(default_factory=ClientCapabilities)


class JobConstraints(BaseModel):
    required_tags: list[str] = Field(default_factory=list)
    required_environments: list[str] = Field(default_factory=list)
    gpu_required: bool = False


class JobSpec(BaseModel):
    action: str
    argv: list[str]
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    constraints: JobConstraints = Field(default_factory=JobConstraints)


class JobCommand(BaseModel):
    command_id: str
    job_id: str
    command: str
    created_at: datetime


class ClientCommand(BaseModel):
    command_id: str
    command: str
    created_at: datetime


class JobRecord(BaseModel):
    job_id: str
    status: str
    assigned_client_id: str | None = None
    spec: JobSpec
    pending_commands: list[JobCommand] = Field(default_factory=list)


class RegisterClientResponse(BaseModel):
    client: dict[str, Any]


class QueueSettings(BaseModel):
    max_concurrent_jobs: int = 1
    enforce_single_pooltool_job: bool = True


class PollResponse(BaseModel):
    assignments: list[JobRecord] = Field(default_factory=list)
    commands: list[JobCommand] = Field(default_factory=list)
    client_commands: list[ClientCommand] = Field(default_factory=list)
    queue_settings: QueueSettings = Field(default_factory=QueueSettings)
    client_max_concurrency: int | None = None
    suggested_poll_delay_seconds: float | None = None


class JobUpdateRequest(BaseModel):
    client_id: str | None = None
    status: str
    exit_code: int | None = None
    message: str | None = None
    metadata_patch: dict[str, Any] = Field(default_factory=dict)


class JobLogChunkRequest(BaseModel):
    client_id: str | None = None
    lines: list[str] = Field(default_factory=list)
