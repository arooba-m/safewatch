# 🚧 SafeWatch v3
### AI-Powered Workplace Safety Monitoring System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Custom_Trained-orange)

## 🎯 What It Does
SafeWatch automatically detects PPE compliance violations in workplace images using a multi-layer AI pipeline.

## 🏗️ Architecture
```
Image Upload → YOLOv8 Detection → RAG OSHA Lookup → 
Multi-Agent AI (LangGraph + CrewAI) → Safety Report
```

## ⚡ Tech Stack
| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (custom trained, 5,269 images) |
| LLM Orchestration | LangGraph + CrewAI (4 agents) |
| RAG Pipeline | ChromaDB + sentence-transformers |
| Backend | FastAPI + JWT Auth + Rate Limiting |
| Database | SQLite |
| Experiment Tracking | MLflow |
| Deployment | Docker + Render |

## 🤖 Multi-Agent Pipeline
1. **Vision Analyst** — reads YOLO detections
2. **Safety Inspector** — checks OSHA regulations via RAG
3. **Report Writer** — generates structured JSON report
4. **Action Recommender** — suggests corrective actions

## 🚀 Quick Start
```bash
git clone https://github.com/arooba-m/safewatch
cd safewatch
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open `http://localhost:8000/dashboard`

## 📊 Features
- ✅ Custom trained PPE detection model
- ✅ Real-time OSHA compliance checking via RAG
- ✅ 4-agent AI pipeline with LangGraph orchestration
- ✅ JWT authentication + rate limiting
- ✅ MLflow experiment tracking
- ✅ Email alerts for violations
- ✅ Interactive dashboard with charts
- ✅ Docker containerization

## 📈 Model Performance
- Dataset: 5,269 labeled construction site images
- Classes: helmet, head (no helmet), person
- Training: 50 epochs on Google Colab T4 GPU
- Framework: YOLOv8n (transfer learning)
```
