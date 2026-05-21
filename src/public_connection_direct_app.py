from __future__ import annotations

from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from public_connection_client_agent import ClientAgent
from public_connection_models import ClientCommand, JobCommand, JobRecord


def create_direct_app(agent: ClientAgent) -> FastAPI:
    app = FastAPI(title="Pattern Client Direct Mode")

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "client_id": agent.client_id,
                "display_name": agent.display_name,
                "mode": "direct",
                "current_jobs": agent.current_jobs,
            }
        )

    @app.post("/api/v1/direct/jobs/start")
    async def start_job(job: JobRecord, background_tasks: BackgroundTasks) -> JSONResponse:
        background_tasks.add_task(agent._start_job, job)
        return JSONResponse({"ok": True, "job_id": job.job_id})

    @app.post("/api/v1/direct/jobs/{job_id}/command")
    async def send_command(job_id: str, command: JobCommand) -> JSONResponse:
        if command.job_id != job_id:
            command = command.model_copy(update={"job_id": job_id})
        agent._handle_commands([command])
        return JSONResponse({"ok": True, "job_id": job_id, "command_id": command.command_id})

    @app.post("/api/v1/direct/client/command")
    async def send_client_command(command: ClientCommand) -> JSONResponse:
        should_shutdown = agent._handle_client_commands([command])
        return JSONResponse(
            {
                "ok": True,
                "command_id": command.command_id,
                "shutdown_requested": should_shutdown,
            }
        )

    return app
