# client_server_connection

`client_server_connection` is the reusable transport layer for running Pattern Learning clients against a central server.

## Features

- Defines shared client/server transport models
- Implements the client agent loop
- Supports direct client mode with a local HTTP app
- Supports public-access and proxy helpers for exposing clients or server endpoints
- Separates connection logic from experiment and application code

## Layout

- `src/public_connection_models.py`: shared protocol models
- `src/public_connection_client_agent.py`: client agent implementation
- `src/public_connection_direct_app.py`: direct-mode FastAPI app
- `src/public_connection_public_access.py`: public access controller
- `src/public_connection_public_proxy.py`: proxy app

This repo is intended to be consumed as a standalone dependency and as a submodule inside the main Pattern Learning repository.
