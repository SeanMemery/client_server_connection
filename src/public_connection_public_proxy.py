from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import Iterable

import httpx
from fastapi import FastAPI, Request, Response


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}


def _filter_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers:
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        filtered[key] = value
    return filtered


def create_app(
    *,
    target_url: str,
    request_timeout_seconds: float = 600.0,
    retry_attempts: int = 5,
) -> FastAPI:
    normalized_target_url = str(target_url).rstrip("/")
    timeout = httpx.Timeout(
        connect=30.0,
        read=max(30.0, float(request_timeout_seconds)),
        write=max(30.0, float(request_timeout_seconds)),
        pool=30.0,
    )
    retry_attempts = max(1, int(retry_attempts))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            base_url=normalized_target_url,
            timeout=timeout,
            follow_redirects=False,
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(title="Pattern Server Public Proxy", lifespan=lifespan)

    async def _proxy_request(request: Request, upstream_path: str) -> Response:
        client: httpx.AsyncClient = request.app.state.client
        body = await request.body()
        headers = _filter_headers(request.headers.items())
        query_string = request.url.query
        target_path = upstream_path if upstream_path.startswith("/") else f"/{upstream_path}"
        if query_string:
            target_path = f"{target_path}?{query_string}"

        last_error: Exception | None = None
        for attempt in range(retry_attempts):
            try:
                upstream = await client.request(
                    request.method,
                    target_path,
                    content=body,
                    headers=headers,
                )
                # Transient upstream overloads occasionally return 5xx for
                # large fetches (e.g. input bundles). Retry safe methods
                # instead of surfacing immediate failure to remote clients.
                if (
                    request.method.upper() in {"GET", "HEAD", "OPTIONS"}
                    and upstream.status_code in {500, 502, 503, 504}
                    and attempt + 1 < retry_attempts
                ):
                    await asyncio.sleep(min(5.0, 0.5 * (attempt + 1)))
                    continue
                return Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    headers=_filter_headers(upstream.headers.items()),
                    media_type=upstream.headers.get("content-type"),
                )
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as error:
                last_error = error
                if attempt + 1 >= retry_attempts:
                    break
                await asyncio.sleep(min(5.0, 0.5 * (attempt + 1)))

        assert last_error is not None
        return Response(
            content=f"Upstream proxy failure: {type(last_error).__name__}: {last_error}",
            status_code=502,
            media_type="text/plain",
        )

    @app.get("/healthz")
    async def healthz() -> Response:
        return Response(content="ok\n", media_type="text/plain")

    @app.api_route("/", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    async def proxy_root(request: Request) -> Response:
        return await _proxy_request(request, "/")

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def proxy_path(path: str, request: Request) -> Response:
        return await _proxy_request(request, f"/{path}")

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--target-url", default="http://127.0.0.1:8791")
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--retry-attempts", type=int, default=2)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_app(
            target_url=args.target_url,
            request_timeout_seconds=args.request_timeout_seconds,
            retry_attempts=args.retry_attempts,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
