# Development and verification

## Environment

The project requires Python 3.12 or newer and uses uv to manage the lock file. The gateway's FastAPI/httpx unit tests do not require a GPU; vLLM is an optional runtime dependency.

```bash
uv sync --extra runtime --extra test
```

When developing only the gateway without loading models, you can omit the `runtime` extra:

```bash
uv sync --extra test
```

## Entry points

```bash
uv run python -m vllm_gateway --help
uv run python -m vllm_gateway
uv run python -m vllm_gateway --preload
```

The `vllm-service` and `vllm-gateway` scripts in `pyproject.toml` both call the same CLI main function.

## Test areas

- `tests/test_catalog.py`: TOML parsing, CLI/environment precedence, command generation, model defaults, and some compatibility tests.
- `tests/test_app.py`: FastAPI routing, stream lease cleanup, management API, and hot reload.
- `tests/test_manager.py`: asynchronous lifecycle, concurrent startup, capacity, draining, crashes, port ownership, and reload invariants.
- `tests/test_script.py`: start/status/load/unload/restart/stop behavior of the repository's `scripts/vllm`, tested with a temporary fake service, and process ownership protection.

Full verification:

```bash
uv run --extra test python -m pytest -q
```

Narrow the loop according to the scope of the change:

```bash
uv run --extra test python -m pytest -q tests/test_catalog.py
uv run --extra test python -m pytest -q tests/test_app.py tests/test_manager.py
uv run --extra test python -m pytest -q tests/test_script.py
```

Package build:

```bash
uv build
```

## CI and Ruff gates

GitHub Actions runs on pushes to `main` and when pull requests are opened or updated, using a Python 3.12 CPU-only runner. CI installs the `test` and `dev` extras, then runs Ruff linting, Ruff format checks, and the full pytest suite in order; it does not install the GPU/CUDA-dependent `runtime` extra.

Run the same checks locally with:

```bash
uv sync --extra test --extra dev
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra test python -m pytest -q
```

After installing pre-commit, commits automatically run Ruff fixes and formatting:

```bash
uv tool install pre-commit
pre-commit install
pre-commit run --all-files
```

The repository does not yet have a type-check gate.

## Change workflow

### Catalog field or command option

1. Add a typed field to `ModelSpec` and handle supported key spellings in the TOML parser.
2. Define the precedence of CLI overrides, model specs, and fallbacks explicitly in `ModelManager._command()`.
3. If vLLM manages the option, add it to the protected flags so that `vllm_args` cannot generate duplicate or conflicting arguments.
4. Update catalog-parsing and command-generation tests, then synchronize `models.toml` and the user documentation.

### Proxy or lifecycle

- Every successful acquire path in the proxy must release its lease, including HTTP errors, SSE close, client cancellation, and unexpected exceptions.
- Manager concurrency tests use `FakeProcess`, `asyncio.Event`, and monkeypatching to control startup/stop without starting real vLLM.
- State-transition changes must cover concurrent first loads, waiter cancellation, active unload, drain timeout, process crashes, and shutdown.
- Do not infer ownership with `fuser`, `pgrep`, port regexes, or a machine-wide process scan.

### Companion script

The companion script version is in `scripts/vllm` in the repository; both `tests/test_script.py` and `test_catalog.py` read this file. Install it to `~/.local/bin/vllm` the first time for deployment:

```bash
install -m 755 scripts/vllm ~/.local/bin/vllm
```

After modifying the script, run `uv run --extra test python -m pytest -q tests/test_script.py` to verify lifecycle and process ownership behavior.

## GPU smoke test

After unit tests pass, only changes to runtime, the catalog, or backend commands require a GPU smoke test:

```bash
vllm restart
vllm load <model>
vllm status
curl http://127.0.0.1:11432/ready
curl http://127.0.0.1:11432/v1/models
```

Then send one non-streaming and one streaming completion, confirming HTTP 200, canonical model routing, and that active requests return to 0 after the stream ends. Model load time and VRAM usage depend on the hardware, so do not treat a GPU smoke test as part of ordinary unit testing.
