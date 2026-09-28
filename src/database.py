import json
import sqlite3
from pathlib import Path


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    inspection_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    location TEXT,
                    note TEXT,
                    image_width INTEGER,
                    image_height INTEGER,
                    detections_json TEXT NOT NULL,
                    severity_json TEXT NOT NULL,
                    annotated_image TEXT,
                    retrieved_context_json TEXT,
                    report_json TEXT
                )
                """
            )
            conn.commit()

    def insert_inspection(self, item):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO inspections (
                    inspection_id,
                    timestamp,
                    image_name,
                    location,
                    note,
                    image_width,
                    image_height,
                    detections_json,
                    severity_json,
                    annotated_image,
                    retrieved_context_json,
                    report_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["inspection_id"],
                    item["timestamp"],
                    item["image_name"],
                    item.get("location", ""),
                    item.get("note", ""),
                    item.get("image_width"),
                    item.get("image_height"),
                    json.dumps(item.get("detections", [])),
                    json.dumps(item.get("severity", {})),
                    item.get("annotated_image", ""),
                    json.dumps(item.get("retrieved_context", [])),
                    json.dumps(item.get("report", {})),
                ),
            )
            conn.commit()

    def list_inspections(self, limit=1000):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    inspection_id,
                    timestamp,
                    image_name,
                    location,
                    severity_json
                FROM inspections
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        output = []

        for row in rows:
            severity = json.loads(row["severity_json"])

            output.append(
                {
                    "inspection_id": row["inspection_id"],
                    "timestamp": row["timestamp"],
                    "image_name": row["image_name"],
                    "location": row["location"] or "",
                    "severity": severity.get("level", "Unknown"),
                }
            )

        return output

    def get_inspection(self, inspection_id):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM inspections
                WHERE inspection_id = ?
                """,
                (inspection_id,),
            ).fetchone()

        if not row:
            return None

        return {
            "inspection_id": row["inspection_id"],
            "timestamp": row["timestamp"],
            "image_name": row["image_name"],
            "location": row["location"] or "",
            "note": row["note"] or "",
            "image_width": row["image_width"],
            "image_height": row["image_height"],
            "detections": json.loads(row["detections_json"]),
            "severity": json.loads(row["severity_json"]),
            "annotated_image": row["annotated_image"] or "",
            "retrieved_context": json.loads(
                row["retrieved_context_json"] or "[]"
            ),
            "report": json.loads(
                row["report_json"] or "{}"
            ),
        }