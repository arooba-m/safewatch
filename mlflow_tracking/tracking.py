import mlflow
import os
from datetime import datetime

MLFLOW_TRACKING_URI = "sqlite:///mlflow_tracking/mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "SafeWatch-PPE-Detection"


def get_or_create_experiment():
    """Get existing experiment or create new one"""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    return mlflow.get_experiment_by_name(EXPERIMENT_NAME)


def log_analysis(
    filename: str,
    username: str,
    detections_count: int,
    helmet_count: int,
    head_count: int,
    avg_confidence: float,
    compliance_score: int,
    severity: str,
    processing_time_ms: float,
    compliance_status: str
):
    """
    Log every analysis to MLflow.
    This creates a permanent record of model performance over time.
    
    Think of it like a lab notebook — every experiment recorded.
    """
    get_or_create_experiment()
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"analysis_{datetime.now().strftime('%H%M%S')}"):

        # Log parameters (what went IN to the analysis)
        mlflow.log_param("filename", filename)
        mlflow.log_param("username", username)
        mlflow.log_param("model", "best.pt")
        mlflow.log_param("compliance_status", compliance_status)
        mlflow.log_param("severity", severity)

        # Log metrics (what came OUT of the analysis)
        mlflow.log_metric("detections_count", detections_count)
        mlflow.log_metric("helmet_count", helmet_count)
        mlflow.log_metric("head_count", head_count)
        mlflow.log_metric("avg_confidence", avg_confidence)
        mlflow.log_metric("compliance_score", compliance_score)
        mlflow.log_metric("processing_time_ms", processing_time_ms)

        # Log violation flag (1 = violation, 0 = compliant)
        violation_flag = 1 if head_count > 0 else 0
        mlflow.log_metric("violation_detected", violation_flag)

    print(f"✅ MLflow: logged analysis for {filename}")