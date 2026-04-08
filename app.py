"""
MisinfoPatrolEnv — FastAPI Server
Exposes the OpenEnv interface over HTTP for HuggingFace Spaces deployment.

Endpoints
---------
POST /reset              — Start a new episode (optional ?task_id=...)
POST /step/{session_id}  — Submit an action
GET  /state/{session_id} — Inspect current state
GET  /tasks              — List available tasks
GET  /health             — Health check (used by HF Space ping)
GET  /                   — Environment info
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from environment import Action, MisinfoPatrolEnv, Observation, RewardInfo
from tasks import TASKS

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MisinfoPatrolEnv",
    description=(
        "OpenEnv: AI Fact-Checker for Viral Social Media Posts. "
        "An RL environment where an agent learns to detect misinformation "
        "by extracting claims, assigning verdicts, and labelling posts."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store  {session_id: MisinfoPatrolEnv}
_sessions: Dict[str, MisinfoPatrolEnv] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Environment metadata")
def root() -> Dict[str, Any]:
    return {
        "name": "MisinfoPatrolEnv",
        "version": "1.0.0",
        "description": "RL environment for AI-powered misinformation detection",
        "tasks": len(TASKS),
        "domains": ["content_moderation", "fact_checking", "nlp"],
        "openenv_spec": "1.0",
        "docs": "/docs",
    }


@app.get("/health", summary="Health check — returns 200 when ready")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset", summary="Start a new episode", response_model=Dict[str, Any])
def reset(task_id: Optional[str] = Query(default=None, description="Pin a specific task ID")) -> Dict[str, Any]:
    """
    Initialise (or re-initialise) the environment and return the first observation.
    If `task_id` is omitted, a task is chosen at random.
    """
    try:
        env = MisinfoPatrolEnv(task_id=task_id)
        obs: Observation = env.reset()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    session_id = env.state()["session_id"]
    _sessions[session_id] = env

    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post(
    "/step/{session_id}",
    summary="Submit an agent action",
    response_model=Dict[str, Any],
)
def step(session_id: str, action: Action) -> Dict[str, Any]:
    """
    Apply the agent's action and return the next observation, reward, done flag,
    and auxiliary info.
    """
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")

    try:
        obs, reward_info, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation": obs.model_dump(),
        "reward": reward_info.model_dump(),
        "done": done,
        "info": info,
    }


@app.get(
    "/state/{session_id}",
    summary="Return full environment state",
    response_model=Dict[str, Any],
)
def state(session_id: str) -> Dict[str, Any]:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env.state()


@app.get("/tasks", summary="List all available tasks")
def list_tasks() -> list:
    return [
        {
            "id": t["id"],
            "difficulty": t["difficulty"],
            "post_text": t["post_text"],
            "num_claims": len(t["claims"]),
            "overall_label": t["overall_label"],
        }
        for t in TASKS
    ]


# ---------------------------------------------------------------------------
# Entry-point (local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
