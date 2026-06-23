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

[4] - 23, 06, 2026 11:43:37
Prompt: Create a new POST endpoint `detect-person` with form-data file upload and optional `save_detection` boolean, using the person_model with threshold 0.3, returning `PersonDetectionResult`.
Output: Added `detect_persons()` to `app/services/detector.py` (decodes the image, runs the person model on class 0, returns `Detection` boxes). Created `app/api/v1/endpoints/persons.py` with the `POST /detect-person` endpoint and registered it in `app/api/v1/router.py`. Added `PERSON_DETECTION_THRESHOLD` (0.3) and `DETECTION_OUTPUT_DIR` settings to `app/core/config.py`; `save_detection=True` persists the result JSON there. Verified end-to-end via TestClient (200 OK, correct schema, file saved).

[5] - 23, 06, 2026 12:10:25
Prompt: Create the PPE detection module: update SafetyLabel to match CLASS_NAMES in scripts/tracker.py (except person/fall/restricted-zone/unknown), and add a POST `detect-ppe` endpoint using PersonDetectionResult, the PERSON_DETECTION_THRESHOLD, saving to DETECTION_OUTPUT_DIR with filename prefix `ppe_detection_{rest}.{original_extension}`.
Output: Replaced NO_HELMET/NO_VEST in `app/schemas/detection.py` SafetyLabel with the six PPE classes (HARD_HAT, NO_HARD_HAT, MASK, NO_MASK, SAFETY_VEST, NO_SAFETY_VEST). Added `PPE_CLASS_LABELS` map, a shared `_decode_image` helper, and `detect_ppe()` to `app/services/detector.py`. Created `app/api/v1/endpoints/ppe.py` (`POST /detect-ppe`) and registered it in the router; `save_detection=True` saves the uploaded image as `ppe_detection_{detection_id}.{ext}`. Updated `safety_monitor.py` warning labels to the new "No ..." classes. Verified end-to-end via TestClient (200 OK, correct schema, file saved with prefix).
