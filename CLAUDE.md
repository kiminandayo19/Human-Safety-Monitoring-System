# CLAUDE.md — FastAPI Project

This file defines rules, conventions, and behavior for AI agents (Claude and others) working in this codebase.

---

## 🔴 MANDATORY: Agent Changelog

**Every time an AI agent is used to make changes or generate output in this project, it MUST update `changelogs/agent-changelogs.md` before finishing.**

### Format

```
[number] - DD, MM, YYYY HH:mm:ss
Prompt: {user prompt}
Output: {output summary}
```

### Rules

- `[number]` is a sequential integer, starting from `1`, incrementing by 1 for each entry. Read the existing file to determine the next number.
- The timestamp uses the **local system time** at the time of the change, in `DD, MM, YYYY HH:mm:ss` format (e.g., `22, 06, 2025 14:35:01`).
- `Prompt` is the **exact or closely paraphrased** user prompt that triggered this agent session.
- `Output` is a **concise summary** (1–3 sentences) of what was done: files created, modified, deleted, or commands run.
- Append entries at the **bottom** of the file. Never reorder or delete existing entries.
- If `changelogs/agent-changelogs.md` does not exist, create it (along with the `changelogs/` directory) before appending.

### Example

```
[1] - 22, 06, 2025 14:35:01
Prompt: Create a user authentication endpoint with JWT
Output: Created `routers/auth.py` with `/login` and `/register` endpoints. Added `models/user.py` and updated `main.py` to include the auth router. JWT logic implemented via `python-jose`.

[2] - 22, 06, 2025 15:10:44
Prompt: Add unit tests for the auth router
Output: Created `tests/test_auth.py` with 6 test cases covering login success, login failure, registration, and token validation using `pytest` and `httpx.AsyncClient`.
```

---

## Project Structure

```
.
app/
├── main.py                  # App factory + lifespan (loads model on boot)
├── core/
│   ├── config.py            # Settings (env / .env)
│   └── logging.py           # Logging setup
├── api/
│   ├── deps.py              # Shared dependencies
│   └── v1/
│       ├── router.py        # Aggregates v1 routers
│       └── endpoints/
│           ├── health.py        # /health, /ready
│           ├── detections.py    # /detections
│           └── monitoring.py    # /monitoring/evaluate, /monitoring/alerts
├── schemas/                 # Pydantic models (detection, monitoring)
└── services/
    ├── detector.py          # CV model wrapper (stub — replace me)
    └── safety_monitor.py    # Rules: detections -> events -> alerts
├── utils/
│   └── storage.py           # Persistence helpers (save detection artifacts)
tests/                       # Pytest smoke tests
changelogs/
│   └── agent-changelogs.md  # ← AI agent must update this every session
CLAUDE.md
Dockerfile
pyproject.toml
README.md
requirements.txt
```

---

## Code Conventions

### General

- Python **3.11+**
- Use **type hints** on all function signatures (parameters and return types)
- Follow **PEP 8**; line length max **100 characters**
- Prefer **async/await** for all route handlers and DB calls

### FastAPI Patterns

- Define routes under `app/api/v1/endpoints/`, one file per domain (e.g., `detections.py`, `monitoring.py`)
- Aggregate all v1 routes in `app/api/v1/router.py` via `router.include_router(...)`
- Mount the v1 router in `app/main.py` under the `/api/v1` prefix
- Use `Depends()` for all shared logic defined in `app/api/deps.py`: model access, current user, permissions
- Always use **Pydantic schemas** from `app/schemas/` for request bodies and response models — never return raw objects
- Use `response_model=` on every endpoint
- Use the **lifespan** context manager in `main.py` to load heavy resources (e.g., CV model) once on boot

```python
# ✅ Good — versioned, typed, response_model set
@router.get("/detections/{id}", response_model=schemas.DetectionOut)
async def get_detection(id: int, detector: Detector = Depends(get_detector)):
    ...

# ❌ Avoid — flat structure, missing response_model, no type hints
@app.get("/detections/{id}")
def get_detection(id, detector=Depends(get_detector)):
    ...
```

### Services

- `app/services/detector.py` — wraps the CV model; treat as a stub until a real model is plugged in. Keep the interface stable: `detect(image) -> list[Detection]`
- `app/services/safety_monitor.py` — pure business logic: consumes detections, applies rules, emits events and alerts. No FastAPI imports here
- Services are instantiated once (at lifespan startup) and injected via `app/api/deps.py`
- Keep services free of HTTP concerns — no `Request`, no `HTTPException`

### Error Handling

- Raise `HTTPException` with appropriate status codes in routers
- Define custom exception handlers in `main.py` for domain-level errors
- Never let raw exceptions propagate to the client

### Environment & Config

- All config via `pydantic-settings` in `core/config.py`
- Load from `.env`; never hardcode secrets
- Commit `.env.example` with placeholder values, never `.env`

---

## Testing

- Use **pytest** with **httpx.AsyncClient** for integration tests
- Mirror the `app/api/v1/endpoints/` structure under `tests/` (e.g., `tests/test_detections.py`, `tests/test_monitoring.py`)
- Always test `/health` and `/ready` as smoke tests
- Aim for coverage of: happy path, validation errors, service-layer errors, edge cases on detection payloads
- Use fixtures for test client initialization and service mocking

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=term-missing
```

---

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Start dev server
uvicorn app.main:app --reload

# API docs available at
http://localhost:8000/docs

# Run via Docker
docker build -t fastapi-app .
docker run -p 8000:8000 fastapi-app
```

---

## Dependencies

Keep `requirements.txt` (and `pyproject.toml` if used) up to date. Core stack:

```
fastapi
uvicorn[standard]
pydantic-settings
httpx                       # async HTTP + test client
pytest
pytest-asyncio
```

CV / model dependencies go here once `services/detector.py` is implemented (e.g., `opencv-python`, `torch`, `ultralytics`).

---

## What AI Agents Should NOT Do

- **Do not** commit `.env` or any file containing real secrets
- **Do not** skip updating `changelogs/agent-changelogs.md`
- **Do not** delete or reorder existing changelog entries
- **Do not** return raw service objects directly from endpoints — always go through a schema
- **Do not** add HTTP/FastAPI imports inside `app/services/` — keep services framework-agnostic
- **Do not** hardcode model paths or thresholds — use `app/core/config.py`
- **Do not** add new top-level directories without updating this file's Project Structure section
- **Do not** modify `services/detector.py`'s public interface without updating `deps.py` and affected endpoints

---

## Checklist Before Finishing Any Agent Session

- [ ] Code follows the conventions in this file
- [ ] Tests added or updated where relevant
- [ ] No secrets or `.env` files staged for commit
- [ ] `changelogs/agent-changelogs.md` updated with this session's entry
