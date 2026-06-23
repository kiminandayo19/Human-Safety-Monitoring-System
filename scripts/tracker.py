import cv2
from pathlib import Path
from ultralytics import YOLO


CLASS_NAMES = [
    "Hard Hat", "No Hard Hat", "Mask",
    "No Mask", "Safety Vest", "No Safety Vest"
]

VIOLATION_CLASSES = {1, 3, 5}  # No Hard Hat, No Mask, No Safety Vest
MODEL_PATH = Path(__file__).parents[1] / "models" / "yolo_50.pt"

model = YOLO(MODEL_PATH)
PERSON_MODEL = YOLO("yolo11n.pt")  # auto-download jika belum ada

# --- ID Switch Rate State ---
_track_history: dict[tuple, int] = {}  # (bbox_hash, cls_idx) -> last_track_id
_id_switches: int = 0
_total_frames: int = 0


def _bbox_hash(x1: int, y1: int, x2: int, y2: int) -> tuple:
    """Approximate spatial identity — grid cell 50px"""
    return (x1 // 50, y1 // 50, x2 // 50, y2 // 50)


def get_id_switch_rate() -> dict:
    if _total_frames == 0:
        return {"id_switches": 0, "total_frames": 0, "rate": "0.00%"}
    return {
        "id_switches": _id_switches,
        "total_frames": _total_frames,
        "rate": f"{(_id_switches / _total_frames) * 100:.2f}%",
    }


def reset_metrics():
    global _track_history, _id_switches, _total_frames
    _track_history = {}
    _id_switches = 0
    _total_frames = 0


def run_tracking(source, output_path=None):
    """
    source: path ke video file (str) atau 0 untuk webcam
    output_path: path output video. None = tidak disimpan

    Yields annotated frame (numpy array) setiap iterasi.
    """
    global _total_frames, _id_switches, _track_history

    writer = None

    try:
        for result in model.track(source, tracker="bytetrack.yaml", persist=True, stream=True):
            frame = result.orig_img.copy()

            # --- Person detection (dual model) ---
            person_results = PERSON_MODEL(frame, classes=[0], verbose=False)
            if person_results[0].boxes is not None:
                for box in person_results[0].boxes.xyxy:
                    px1, py1, px2, py2 = map(int, box)
                    cv2.rectangle(frame, (px1, py1),
                                  (px2, py2), (255, 165, 0), 2)
                    cv2.putText(frame, "Person", (px1, py1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)

            # --- PPE detection (model utama) ---
            if result.boxes is not None and result.boxes.id is not None:
                for box, track_id, cls_idx, conf in zip(
                    result.boxes.xyxy,
                    result.boxes.id,
                    result.boxes.cls,
                    result.boxes.conf,
                ):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_id)
                    cls_idx = int(cls_idx)
                    conf = float(conf)

                    # --- Draw box & label ---
                    is_violation = cls_idx in VIOLATION_CLASSES
                    color = (0, 0, 255) if is_violation else (0, 255, 0)
                    label = f"ID{track_id} {CLASS_NAMES[cls_idx]} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    # --- ID Switch tracking ---
                    key = (_bbox_hash(x1, y1, x2, y2), cls_idx)
                    if key in _track_history and _track_history[key] != track_id:
                        _id_switches += 1
                    _track_history[key] = track_id

            # --- Update frame counter & overlay metrics ---
            _total_frames += 1
            rate_info = get_id_switch_rate()
            cv2.putText(
                frame,
                f"ID Switches: {rate_info['id_switches']} | Rate: {
                    rate_info['rate']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )

            # --- Setup writer di frame pertama ---
            if output_path and writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    30, (w, h),
                )

            if writer:
                writer.write(frame)

            cv2.imshow("PPE Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            yield frame

    except Exception as e:
        print(e)
    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"done — {get_id_switch_rate()}")


if __name__ == "__main__":
    reset_metrics()
    for frame in run_tracking(
        source=Path(__file__).parents[1] / "hardhat.mp4",
        output_path=str(Path(__file__).parents[1] / "results" / "output.mp4"),
    ):
        pass
