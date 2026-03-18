import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from agents.orchestrator import run_safety_analysis

# Simulate what YOLO would detect
test_detections = [
    {"label": "head", "confidence": 0.87, "bbox": [100, 100, 200, 200]},
    {"label": "person", "confidence": 0.92, "bbox": [80, 80, 220, 400]}
]

test_violations = ["head detected - no helmet"]
test_status = "NON-COMPLIANT"

print("Running full agent pipeline...\n")
result = run_safety_analysis(test_detections, test_violations, test_status)

print("\n=== RESULTS ===")
print("Vision Summary:", result["vision_summary"][:200])
print("Severity:", result["inspection_result"].get("severity"))
print("Compliance Score:", result["inspection_result"].get("compliance_score"))
print("Actions:", result["actions"])