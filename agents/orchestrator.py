from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import json
import re

# ── State — this is passed between all nodes ──────────────
class SafetyState(TypedDict):
    """
    Think of this like a clipboard passed between agents.
    Each agent reads from it and adds their findings to it.
    """
    detections: List[dict]        # raw YOLO detections
    violations: List[str]         # list of violations found
    compliance_status: str        # COMPLIANT / NON-COMPLIANT
    osha_context: str             # retrieved from ChromaDB
    vision_summary: str           # Agent 1 output
    inspection_result: dict       # Agent 2 output
    final_report: dict            # Agent 3 output
    actions: List[str]            # Agent 4 output
    error: str                    # if something goes wrong


# ── Node Functions (each is one step in the graph) ────────

def retrieve_osha_node(state: SafetyState) -> SafetyState:
    """
    Step 1: Before agents run, retrieve relevant OSHA rules
    from ChromaDB based on what violations were found
    """
    from rag.retriever import retrieve_osha_context

    violations = state.get("violations", [])
    query = f"PPE violations: {', '.join(violations)}" if violations else "construction site PPE requirements"

    print(f"RAG: Retrieving OSHA context...")
    osha_context = retrieve_osha_context(query, k=3)

    return {**state, "osha_context": osha_context}


def vision_analysis_node(state: SafetyState) -> SafetyState:
    """
    Step 2: Analyze what YOLO detected
    """
    from langchain_ollama import OllamaLLM
    import os

    llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3.2"))

    detections = state.get("detections", [])
    detection_text = "\n".join([
        f"- {d['label']} (confidence: {d['confidence']*100:.0f}%)"
        for d in detections
    ]) or "No objects detected"

    prompt = f"""You are a computer vision analyst.
    
Analyze these workplace detection results:
{detection_text}

In 2-3 sentences describe:
1. How many workers are visible
2. What safety equipment is present
3. What safety equipment is missing"""

    print("Agent 1: Vision Analyst running...")
    summary = llm.invoke(prompt)

    return {**state, "vision_summary": summary}


def safety_inspection_node(state: SafetyState) -> SafetyState:
    """
    Step 3: Check compliance against OSHA rules
    """
    from langchain_ollama import OllamaLLM
    import os

    llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3.2"))
    prompt = f"""You are a certified OSHA safety inspector.

Vision Analysis:
{state.get('vision_summary', '')}

Violations Found:
{', '.join(state.get('violations', [])) or 'None'}

OSHA Regulations (retrieved from database):
{state.get('osha_context', '')}

Respond in JSON only:
{{
    "severity": "CRITICAL or HIGH or MEDIUM or LOW",
    "compliance_score": number 0-100,
    "osha_standards_violated": ["list of standards"],
    "estimated_fine": "dollar amount"
}}"""

    print("Agent 2: Safety Inspector running...")
    result_text = llm.invoke(prompt)

    # Extract JSON safely
    try:
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        result = json.loads(match.group()) if match else {}
    except Exception:
        result = {
            "severity": "MEDIUM",
            "compliance_score": 50,
            "osha_standards_violated": [],
            "estimated_fine": "Unknown"
        }

    return {**state, "inspection_result": result}


def report_writing_node(state: SafetyState) -> SafetyState:
    """
    Step 4: Write the formal safety report
    """
    from langchain_ollama import OllamaLLM
    import os

    llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3.2"))

    inspection = state.get("inspection_result", {})

    prompt = f"""You are a safety report writer.

Write a professional safety report as JSON only:
{{
    "executive_summary": "2 sentence summary",
    "severity": "{inspection.get('severity', 'MEDIUM')}",
    "compliance_score": {inspection.get('compliance_score', 50)},
    "violations_detail": ["detailed violation descriptions"],
    "osha_violations": {json.dumps(inspection.get('osha_standards_violated', []))},
    "estimated_fine": "{inspection.get('estimated_fine', 'Unknown')}",
    "immediate_actions": ["3 specific immediate actions"]
}}

Output ONLY valid JSON."""

    print("Agent 3: Report Writer running...")
    report_text = llm.invoke(prompt)

    try:
        match = re.search(r'\{.*\}', report_text, re.DOTALL)
        report = json.loads(match.group()) if match else {}
    except Exception:
        report = {
            "executive_summary": state.get("vision_summary", ""),
            "severity": inspection.get("severity", "MEDIUM"),
            "compliance_score": inspection.get("compliance_score", 50),
            "violations_detail": state.get("violations", []),
            "osha_violations": [],
            "estimated_fine": inspection.get("estimated_fine", "Unknown"),
            "immediate_actions": []
        }

    return {**state, "final_report": report}


def action_recommendation_node(state: SafetyState) -> SafetyState:
    """
    Step 5: Generate corrective actions
    """
    from langchain_ollama import OllamaLLM
    import os

    llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3.2"))

    prompt = f"""You are a safety compliance advisor.

Based on:
- Violations: {', '.join(state.get('violations', [])) or 'None'}
- Severity: {state.get('inspection_result', {}).get('severity', 'MEDIUM')}

Give exactly 3 immediate actions (do today) as a simple list.
Be specific and practical. No JSON needed, just a numbered list."""

    print("Agent 4: Action Recommender running...")
    actions_text = llm.invoke(prompt)
    actions = [
        line.strip()
        for line in actions_text.split('\n')
        if line.strip() and line.strip()[0].isdigit()
    ]

    return {**state, "actions": actions}


def should_continue(state: SafetyState) -> str:
    """
    Router function — LangGraph calls this to decide
    what to do next. If error, go to END. Otherwise continue.
    """
    if state.get("error"):
        return "end"
    return "continue"


# ── Build the Graph ────────────────────────────────────────

def build_safety_graph():
    """
    This builds the state machine.
    Think of it like drawing a flowchart in code.
    """
    graph = StateGraph(SafetyState)

    # Add nodes (each node = one step)
    graph.add_node("retrieve_osha", retrieve_osha_node)
    graph.add_node("vision_analysis", vision_analysis_node)
    graph.add_node("safety_inspection", safety_inspection_node)
    graph.add_node("report_writing", report_writing_node)
    graph.add_node("action_recommendation", action_recommendation_node)

    # Add edges (define the flow between nodes)
    graph.set_entry_point("retrieve_osha")
    graph.add_edge("retrieve_osha", "vision_analysis")
    graph.add_edge("vision_analysis", "safety_inspection")
    graph.add_edge("safety_inspection", "report_writing")
    graph.add_edge("report_writing", "action_recommendation")
    graph.add_edge("action_recommendation", END)

    return graph.compile()


# ── Main function called by FastAPI ───────────────────────

def run_safety_analysis(detections: list, violations: list, compliance_status: str) -> dict:
    """
    Entry point — called from main.py
    Runs the full LangGraph pipeline
    """
    print("\n Starting LangGraph Safety Analysis Pipeline...")

    # Build the graph
    graph = build_safety_graph()

    # Initial state
    initial_state = SafetyState(
        detections=detections,
        violations=violations,
        compliance_status=compliance_status,
        osha_context="",
        vision_summary="",
        inspection_result={},
        final_report={},
        actions=[],
        error=""
    )

    # Run the graph
    final_state = graph.invoke(initial_state)

    print("Pipeline complete!\n")

    return {
        "vision_summary": final_state.get("vision_summary", ""),
        "inspection_result": final_state.get("inspection_result", {}),
        "final_report": final_state.get("final_report", {}),
        "actions": final_state.get("actions", []),
        "osha_context_used": final_state.get("osha_context", "")[:200]
    }