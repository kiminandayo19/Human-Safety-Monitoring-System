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

[6] - 23, 06, 2026 12:22:41
Prompt: Change the save filename template to `{detection_type}_detection_{rest}.{original_extension}`, saving the image with OpenCV-drawn annotations plus a matching `.json` response file; and combine `detect-ppe`/`detect-person` into a single API differentiated by a `detection_type: ppe | person` payload.
Output: Added `DetectionType` enum to `app/schemas/detection.py`. Refactored `app/services/detector.py` into a unified `detect(detection_type, ...)` method (plus `_label_for`) and added an `annotate()` helper that draws boxes/labels via OpenCV and re-encodes. Replaced `persons.py`/`ppe.py` with `app/api/v1/endpoints/detect.py` (`POST /detect`) and updated the router; `save_detection=True` now writes both `{type}_detection_{id}.{ext}` (annotated image) and `.json` (response) to `DETECTION_OUTPUT_DIR`. Verified via TestClient: both types 200 OK, invalid type 422, both files saved.

[7] - 23, 06, 2026 12:26:02
Prompt: Move the `_save_detection` method in endpoints/detect.py to a utility instead of there.
Output: Created `app/utils/storage.py` with `save_detection_result(...)` (the former `_save_detection` body) and an `app/utils/__init__.py`. Updated `app/api/v1/endpoints/detect.py` to import and call it, removing the local function and now-unused imports. Documented the new `app/utils/` directory in CLAUDE.md's Project Structure. Verified via TestClient (200 OK, both artifacts saved).

[8] - 23, 06, 2026 13:11:01
Prompt: Create a Person tracking module with a `track-person` endpoint: track persons (person model) + per-person PPE via ByteTrack, draw bbox with meta (person_id, helmet, vest, mask), add ID Switch Rate consistency, show FPS top-left, output an annotated video + JSON tracking file.
Output: Added `app/schemas/tracking.py` (TrackedPerson/TrackingFrame/IDSwitchRate/TrackingData/PersonTrackingResult). Implemented `_IDSwitchTracker` and `track_persons()` (+ `_ppe_status_for_person`, `_track_persons_in_frame`) in `app/services/detector.py`: per-frame ByteTrack on both models, PPE-to-person association by box center, FPS overlay, annotated MP4 writing. Added `tracking_output_paths()`/`save_tracking_json()` to `app/utils/storage.py`. Created `app/api/v1/endpoints/track.py` (`POST /track-person`) and registered it in the router. Verified end-to-end via TestClient on sample clips (200 OK; person ids, meta, ID switch rate, FPS, video + JSON all produced).

[9] - 23, 06, 2026 13:30:29
Prompt: Generate a single .tex technical report covering data, model, evaluation, system design, AI usage log (primary, 3-5), and limitations.
Output: Created `technical_report.tex` (self-contained LaTeX article) with the six requested sections, grounded in the repo (PPE/person models, class taxonomy, ByteTrack, ID Switch Rate, FPS, layered FastAPI design) and five primary changelog entries (#3,4,5,6,8). Verified it compiles cleanly with pdflatex to a PDF.

[10] - 23, 06, 2026 13:44:46
Prompt: Adjust the README.md.
Output: Rewrote `README.md` to reflect the current system (no longer a stub boilerplate): real two-model setup, updated project structure (detect.py, track.py, utils/storage.py, tracking schemas), a Models section with the PPE class taxonomy and config keys, the `/detect` and `/track-person` endpoints with curl examples, and an output-artifacts section.
