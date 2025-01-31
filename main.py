import sqlite3
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import Dict
import sqlite3
from contextlib import closing
from aggregate_ocean import statistically_scored_ocean



import numpy as np
import pandas as pd

class TextRequest(BaseModel):
    text: str


class ProcessResponse(BaseModel):
    pid: str
    status: str
    individual_scores: Dict[str, float] = {}
    aggregated_score: float = 0.0
    text: str

app = FastAPI()


def init_db():
    with sqlite3.connect("processes.db") as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processes (
                    pid TEXT PRIMARY KEY,
                    status TEXT,
                    individual_scores TEXT,
                    aggregated_score REAL,
                    text TEXT
                )
            """)
            conn.commit()


def insert_process(pid: str, text: str):
    with sqlite3.connect("processes.db") as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(
                "INSERT INTO processes (pid, status, individual_scores, aggregated_score, text) VALUES (?, ?, ?, ?, ?)",
                (pid, "Processing", "{}", 0.0, text)
            )
            conn.commit()


def update_process(pid: str, individual_scores: Dict[str, float], aggregated_score: float):
    with sqlite3.connect("processes.db") as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(
                "UPDATE processes SET status = ?, individual_scores = ?, aggregated_score = ? WHERE pid = ?",
                ("Completed", str(individual_scores), aggregated_score, pid)
            )
            conn.commit()


def get_process_status(pid: str):
    with sqlite3.connect("processes.db") as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT * FROM processes WHERE pid = ?", (pid,))
            row = cursor.fetchone()
            if row:
                return {
                    "pid": row[0],
                    "status": row[1],
                    "individual_scores": eval(row[2]),
                    "aggregated_score": row[3],
                    "text": row[4]
                }
            return None


async def process_text_background(pid: str, text: str):
    try:
        scores = statistically_scored_ocean(text)
        individual_scores = scores
        aggregated_score = np.mean(list(scores.values()))
        update_process(pid, individual_scores, aggregated_score)
    except Exception as e:
        with sqlite3.connect("processes.db") as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute(
                    "UPDATE processes SET status = ?, individual_scores = ? WHERE pid = ?",
                    ("Failed", str({"error": str(e)}), pid)
                )
                conn.commit()

# FastAPI endpoint - starts background task
@app.post("/process-text/", response_model=ProcessResponse)
async def process_text(request: TextRequest, background_tasks: BackgroundTasks):
    pid = str(uuid.uuid4())  # Generate a unique ID
    text = request.text
    insert_process(pid, text)
    background_tasks.add_task(process_text_background, pid, text)  # Add background task
    return {"pid": pid, "status": "Processing", "individual_scores": {},"text": text}

# Endpoint to get process status and scores by PID
@app.get("/get-scores/{pid}", response_model=ProcessResponse)
async def get_scores(pid: str):
    process = get_process_status(pid)
    if not process:
        raise HTTPException(status_code=404, detail="Process not found")
    return process

@app.on_event("startup")
async def startup_event():
    init_db()
