# Agent Changelogs

This file is maintained automatically by AI agents working in this project.
Every agent session must append an entry here. See `CLAUDE.md` for the format rules.

---

[1] - 22, 06, 2026 17:55:00
Prompt: Create a CLAUDE.md files for FastAPI project. Add rules that each time the AI was used, they need to edit the `changelogs/agent-changelogs.md` in format [number] - dd, mm, yyyy HH:mm:ss / Prompt: {user prompt} / Output: {output summary}
Output: Created `CLAUDE.md` with FastAPI project conventions and a mandatory agent changelog rule at the top. Also created `changelogs/agent-changelogs.md` as the initial changelog file with a header and format instructions.

[2] - 22, 06, 2026 17:57:00
Prompt: Change the CLAUDE.md to follow this structures [app/main.py, app/core/, app/api/v1/endpoints/, app/schemas/, app/services/, tests/, changelogs/, CLAUDE.md, Dockerfile, pyproject.toml, README.md, requirements.txt]
Output: Updated the Project Structure section in `CLAUDE.md` to reflect the new `app/`-rooted layout with versioned API routing and CV service stubs. Revised FastAPI Patterns, Services, Testing, Running the App, Dependencies, and Agent Constraints sections to align with the new structure.

[3] - 23, 06, 2026 11:27:11
Prompt: Add model loader. The loader should be done when the service was booting. There will be 2 models: models/yolo_50.pt (PPE Detection) and models/yolo_person_tracker.pt (Raw Pretrained YOLOv11 for Person Detection and tracker).
Output: Added `PPE_MODEL_PATH` and `PERSON_MODEL_PATH` settings to `app/core/config.py` (replacing the single `MODEL_PATH`). Reworked `app/services/detector.py` to load both YOLO models via ultralytics in `load()` (called on boot from the existing lifespan in `main.py`); `is_ready()` now requires both models. Verified both models load and the app imports cleanly.
