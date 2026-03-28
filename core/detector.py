import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""
os.environ["MPLBACKEND"] = "Agg"
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best.pt")
print(f"Model loaded with classes: {model.names}")

# ── Class definitions based on my trained model ─────────
# head   = person WITHOUT helmet → VIOLATION
# helmet = person WITH helmet    → COMPLIANT
# person = full body detected

VIOLATION_CLASSES = ["head"]        # head = no helmet
COMPLIANT_CLASSES = ["helmet"]      # helmet = wearing helmet


def analyze_image(image_bytes: bytes) -> dict:
    """Analyze a single image for PPE compliance"""

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img, conf=0.5)[0]


    detections = []
    labels_found = []

    for box in results.boxes:
        label = model.names[int(box.cls)]
        confidence = float(box.conf)
        bbox = box.xyxy[0].tolist()
        labels_found.append(label)
        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "bbox": bbox
        })

    # ── Compliance Logic ───────────────────────────────────
    # head detected   = worker without helmet = VIOLATION
    # helmet detected = worker with helmet   = COMPLIANT
    violations = []

    head_count = labels_found.count("head")
    helmet_count = labels_found.count("helmet")
    person_count = labels_found.count("person")

    if head_count > 0:
        violations.append(
            f"{head_count} worker(s) detected WITHOUT helmet"
        )

    # Determine overall status
    if not detections:
        status = "NO PERSON DETECTED"
    elif violations:
        status = "NON-COMPLIANT"
    else:
        status = "COMPLIANT"

    # Draw boxes on image
    annotated_img = results.plot()
    _, buffer = cv2.imencode(".jpg", annotated_img)

    return {
        "detections": detections,
        "labels_found": labels_found,
        "compliance_status": status,
        "violations": violations,
        "head_count": head_count,
        "helmet_count": helmet_count,
        "person_count": person_count,
        "annotated_image": buffer.tobytes()
    }


def analyze_with_tracking(image_bytes: bytes) -> dict:
    """Analyze image with worker ID tracking across frames"""

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.track(img, persist=True, conf=0.5)[0]

    detections = []
    labels_found = []
    tracked_workers = []

    for box in results.boxes:
        label = model.names[int(box.cls)]
        confidence = float(box.conf)
        bbox = box.xyxy[0].tolist()
        labels_found.append(label)

        track_id = None
        if results.boxes.id is not None:
            track_id = int(box.id) if box.id is not None else None

        detection = {
            "label": label,
            "confidence": round(confidence, 2),
            "bbox": bbox,
            "worker_id": f"W{track_id:03d}" if track_id else "unknown"
        }
        detections.append(detection)

        if track_id:
            tracked_workers.append({
                "worker_id": f"W{track_id:03d}",
                "label": label,
                "confidence": round(confidence, 2)
            })

    violations = []
    head_count = labels_found.count("head")
    if head_count > 0:
        violations.append(
            f"{head_count} worker(s) detected WITHOUT helmet "
        )

    status = "NON-COMPLIANT" if violations else "COMPLIANT"

    annotated_img = results.plot()
    _, buffer = cv2.imencode(".jpg", annotated_img)

    return {
        "detections": detections,
        "labels_found": labels_found,
        "compliance_status": status,
        "violations": violations,
        "tracked_workers": tracked_workers,
        "annotated_image": buffer.tobytes()
    }