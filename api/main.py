import time
import base64
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import BackgroundTasks
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
try:
    from mlflow_tracking.tracking import log_analysis
    MLFLOW_ENABLED = True
except Exception:
    MLFLOW_ENABLED = False

    def log_analysis(*args, **kwargs):
        pass


load_dotenv()


from core.detector import analyze_image, analyze_with_tracking
from core.database import (
    init_db, save_report, get_reports,

    get_stats, save_worker_tracks
)
from agents.orchestrator import run_safety_analysis
from api.auth import (
    get_current_user, create_token,
    authenticate_user, register_user
)
from api.alerts import send_violation_alert

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("safewatch")

# ── Rate Limiter ───────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="SafeWatch API v3",
    description="""
    AI-Powered Workplace Safety Monitoring System
    
    Features:
    - YOLOv8 PPE Detection (custom trained model)
    - Multi-Agent AI Analysis (LangGraph + CrewAI)
    - RAG-powered OSHA compliance (ChromaDB)
    - JWT Authentication
    - Rate Limiting
    - MLflow Experiment Tracking
    """,
    version="3.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize DB on startup ───────────────────────────────
init_db()
logger.info("SafeWatch v3 started!")

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/dashboard", tags=["Frontend"])
def dashboard():
    return FileResponse("frontend/index.html")
# ══════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════

@app.post("/register", tags=["Auth"])
def register(username: str, password: str):
    """Create a new user account"""
    if len(password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters"
        )
    success = register_user(username, password)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )
    return {"message": f"User '{username}' registered successfully ✅"}


@app.post("/token", tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get JWT token"""
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=401,
            detail="Wrong username or password"
        )
    token = create_token({"sub": form_data.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "message": "Login successful"
    }



@app.post("/analyze", tags=["Analysis"])
@limiter.limit("10/minute")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """
    Main endpoint: Upload image → get full AI safety analysis
    
    Pipeline:
    1. YOLOv8 detection
    2. RAG OSHA lookup
    3. Multi-agent analysis (LangGraph)
    4. Save to database
    5. Send alert if violations found
    """
    start_time = time.time()
    logger.info(f"📸 Analysis request: {file.filename} by {current_user}")

    
    contents = await file.read()

    
    logger.info("Running YOLOv8 detection...")
    detection_result = analyze_image(contents)

    # Step 3: Run agent pipeline
    logger.info("Running agent pipeline...")
    agent_result = run_safety_analysis(
        detections=detection_result["detections"],
        violations=detection_result["violations"],
        compliance_status=detection_result["compliance_status"]
    )

    # Step 4: Calculate metrics
    confidences = [
        d["confidence"] for d in detection_result["detections"]
    ]
    avg_confidence = round(
        sum(confidences) / len(confidences), 2
    ) if confidences else 0

    compliance_score = agent_result.get(
        "final_report", {}
    ).get("compliance_score", 0)

    severity = agent_result.get(
        "inspection_result", {}
    ).get("severity", "UNKNOWN")

    # Step 5: Save to database
    report_id = save_report(
        username=current_user,
        filename=file.filename,
        status=detection_result["compliance_status"],
        violations=detection_result["violations"],
        detections_count=len(detection_result["detections"]),
        confidence_avg=avg_confidence,
        severity=severity,
        compliance_score=compliance_score,
        llm_report=agent_result.get("final_report", {})
    )

    # Step 6: Save worker tracks if any
    if detection_result.get("tracked_workers"):
        save_worker_tracks(
            detection_result["tracked_workers"], report_id
        )

    # Step 7: Send email alert if violations found
    if detection_result["violations"]:
        send_violation_alert(
            violations=detection_result["violations"],
            image_bytes=detection_result["annotated_image"],
            username=current_user,
            compliance_score=compliance_score
        )

    duration = round((time.time() - start_time) * 1000, 2)

    # Log to MLflow
    try:
        log_analysis(
            filename=file.filename,
            username=current_user,
            detections_count=len(detection_result["detections"]),
            helmet_count=detection_result.get("helmet_count", 0),
            head_count=detection_result.get("head_count", 0),
            avg_confidence=avg_confidence,
            compliance_score=compliance_score,
            severity=severity,
            processing_time_ms=duration,
            compliance_status=detection_result["compliance_status"]
        )
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}") 

    logger.info(f"Analysis complete in {duration}ms")
    
    
    

    # Step 8: Return full response
    img_base64 = base64.b64encode(
        detection_result["annotated_image"]
    ).decode("utf-8")

    return JSONResponse({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": file.filename,
        "analyzed_by": current_user,
        "processing_time_ms": duration,
        "detection": {
            "detections": detection_result["detections"],
            "detections_count": len(detection_result["detections"]),
            "helmet_count": detection_result.get("helmet_count", 0),
            "head_count": detection_result.get("head_count", 0),
            "avg_confidence": avg_confidence,
            "compliance_status": detection_result["compliance_status"],
            "violations": detection_result["violations"]
        },
        "ai_analysis": {
            "severity": severity,
            "compliance_score": compliance_score,
            "vision_summary": agent_result.get("vision_summary", ""),
            "final_report": agent_result.get("final_report", {}),
            "immediate_actions": agent_result.get("actions", []),
            "osha_context": agent_result.get("osha_context_used", "")
        },
        "annotated_image": img_base64
    })



@app.get("/reports", tags=["Dashboard"])
def get_all_reports(
    current_user: str = Depends(get_current_user)
):
    """Get all reports for current user"""
    return get_reports(username=current_user)


@app.get("/stats", tags=["Dashboard"])
def get_statistics(
    current_user: str = Depends(get_current_user)
):
    """Get dashboard statistics"""
    stats = get_stats()
    return {
        "total_analyses": stats["total_analyses"],
        "compliant": stats["compliant"],
        "violations": stats["violations"],
        "compliance_rate": f"{round((stats['compliant'] / max(stats['total_analyses'], 1)) * 100, 1)}%",
        "avg_confidence": stats["avg_confidence"],
        "avg_compliance_score": stats["avg_compliance_score"]
    }



@app.get("/", tags=["Health"])
def health_check():
    return {
        "system": "SafeWatch v3",
        "status": "running",
        "features": [
            "YOLOv8 PPE Detection",
            "LangGraph Multi-Agent Pipeline",
            "RAG OSHA Compliance",
            "JWT Authentication",
            "Rate Limiting",
            "MLflow Tracking"
        ]
    }