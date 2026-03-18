import cv2
import numpy as np
from core.detector import analyze_with_tracking, model, VIOLATION_CLASSES
import base64

def process_video(video_bytes: bytes, frame_skip: int = 10) -> dict:
    """
    Process a video file for PPE compliance
    frame_skip: process every Nth frame (10 = process every 10th frame)
    Higher = faster but less accurate
    """

    
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")

    all_detections = []
    all_violations = []
    violation_frames = []
    frame_results = []
    frame_count = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for CPU optimization
        if frame_count % frame_skip != 0:
            continue

        processed_count += 1

        # Convert frame to bytes
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        # Run detection
        result = analyze_with_tracking(frame_bytes)

        all_detections.extend(result["detections"])
        all_violations.extend(result["violations"])

        frame_result = {
            "frame": frame_count,
            "timestamp_sec": round(frame_count / fps, 2) if fps > 0 else 0,
            "detections": result["detections"],
            "violations": result["violations"],
            "status": result["compliance_status"]
        }
        frame_results.append(frame_result)

        # Save violation frames
        if result["violations"]:
            violation_frames.append({
                "frame": frame_count,
                "timestamp_sec": round(frame_count / fps, 2),
                "violations": result["violations"],
                "annotated_image": base64.b64encode(
                    result["annotated_image"]
                ).decode("utf-8")
            })

    cap.release()

    # Clean up temp file
    import os
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Calculate overall stats
    unique_violations = list(set(all_violations))
    total_violation_frames = len(violation_frames)
    compliance_rate = round(
        ((processed_count - total_violation_frames) / max(processed_count, 1)) * 100, 1
    )

    return {
        "video_stats": {
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "duration_seconds": round(duration, 1),
            "fps": round(fps, 1)
        },
        "compliance_rate": compliance_rate,
        "total_violations_found": total_violation_frames,
        "unique_violations": unique_violations,
        "violation_frames": violation_frames[:5],  # first 5 violation frames
        "frame_results": frame_results,
        "summary_status": "NON-COMPLIANT" if unique_violations else "COMPLIANT"
    }