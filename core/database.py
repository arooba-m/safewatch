import sqlite3
from datetime import datetime

DB_NAME = "safewatch.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            username         TEXT UNIQUE NOT NULL,
            hashed_password  TEXT NOT NULL,
            created_at       TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT,
            filename          TEXT,
            username          TEXT,
            compliance_status TEXT,
            violations        TEXT,
            detections_count  INTEGER,
            confidence_avg    REAL,
            severity          TEXT,
            compliance_score  INTEGER,
            llm_report        TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS worker_tracks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            worker_id   TEXT,
            label       TEXT,
            confidence  REAL,
            report_id   INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            endpoint    TEXT,
            username    TEXT,
            status_code INTEGER,
            duration_ms REAL
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized")


def save_report(username, filename, status, violations,
                detections_count, confidence_avg, severity,
                compliance_score, llm_report):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.execute(
        """INSERT INTO reports
           (timestamp, filename, username, compliance_status,
            violations, detections_count, confidence_avg,
            severity, compliance_score, llm_report)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         filename, username, status, str(violations),
         detections_count, confidence_avg,
         severity, compliance_score, str(llm_report))
    )
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id


def save_worker_tracks(tracks: list, report_id: int):
    conn = sqlite3.connect(DB_NAME)
    for t in tracks:
        conn.execute(
            """INSERT INTO worker_tracks
               (timestamp, worker_id, label, confidence, report_id)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             t["worker_id"], t["label"], t["confidence"], report_id)
        )
    conn.commit()
    conn.close()


def get_reports(username=None, limit=50):
    conn = sqlite3.connect(DB_NAME)
    if username:
        rows = conn.execute(
            "SELECT * FROM reports WHERE username=? ORDER BY id DESC LIMIT ?",
            (username, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM reports ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [
        {
            "id": r[0], "timestamp": r[1], "filename": r[2],
            "username": r[3], "compliance_status": r[4],
            "violations": r[5], "detections_count": r[6],
            "confidence_avg": r[7], "severity": r[8],
            "compliance_score": r[9], "llm_report": r[10]
        }
        for r in rows
    ]


def get_stats():
    conn = sqlite3.connect(DB_NAME)
    total = conn.execute(
        "SELECT COUNT(*) FROM reports"
    ).fetchone()[0]
    compliant = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE compliance_status LIKE '%COMPLIANT%' AND compliance_status NOT LIKE '%NON%'"
    ).fetchone()[0]
    violations = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE compliance_status LIKE '%NON%' OR compliance_status LIKE '%REVIEW%'"
    ).fetchone()[0]
    avg_conf = conn.execute(
        "SELECT AVG(confidence_avg) FROM reports"
    ).fetchone()[0]
    avg_score = conn.execute(
        "SELECT AVG(compliance_score) FROM reports"
    ).fetchone()[0]
    conn.close()
    return {
        "total_analyses": total,
        "compliant": compliant,
        "violations": violations,
        "avg_confidence": round(avg_conf or 0, 2),
        "avg_compliance_score": round(avg_score or 0, 2)
    }