"""
MisinfoPatrolEnv — Baseline Inference Script
=============================================
Runs a language-model agent against all three tasks and reports reproducible scores.

Required environment variables
-------------------------------
    API_BASE_URL   The base URL of the deployed MisinfoPatrolEnv HF Space
                   (e.g. https://your-username-misinfopatrolenv.hf.space)
    MODEL_NAME     Model identifier for inference (e.g. meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN       Your Hugging Face API key (used as the OpenAI API key)

Optional
--------
    OPENAI_BASE_URL  Override the OpenAI-compatible API base URL
                     Defaults to the HF Inference API endpoint.

Usage
-----
    API_BASE_URL=https://... MODEL_NAME=... HF_TOKEN=hf_... python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration (read from environment — mandatory per hackathon spec)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

OPENAI_BASE_URL: str = os.environ.get(
    "OPENAI_BASE_URL",
    "https://api-inference.huggingface.co/v1",
)

client = OpenAI(
    api_key=HF_TOKEN or "dummy",
    base_url=OPENAI_BASE_URL,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert misinformation analyst and fact-checker.

Your job: analyse a viral social-media post and produce a structured JSON fact-check report.

Instructions:
1. Read the post carefully.
2. Extract every distinct factual claim (things that are asserted as facts).
3. For each claim, assign a verdict:
   - "true"          — supported by scientific consensus / reliable sources
   - "false"         — directly contradicted by evidence
   - "misleading"    — technically accurate but presented in a deceptive or out-of-context way
   - "unverifiable"  — cannot be confirmed or denied with available information
4. Assign a single overall label to the post:
   - "credible"        — all claims are accurate and not misleading
   - "misinformation"  — contains clearly false claims intended to deceive
   - "misleading"      — technically accurate but deliberately deceptive framing
   - "unverifiable"    — claims cannot be checked
5. Write a concise reasoning string (2–4 sentences).

IMPORTANT: Respond ONLY with a valid JSON object — no markdown fences, no preamble.
The JSON must have exactly these keys:
  "claims"        — array of strings
  "verdicts"      — array of strings (same length as claims)
  "overall_label" — string
  "reasoning"     — string
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON from model output, stripping any markdown fences."""
    text = text.strip()
    # Remove ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _call_model(post_text: str, max_retries: int = 3) -> Dict[str, Any]:
    """Ask the LLM to fact-check a post. Returns parsed action dict."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Fact-check this viral post:\n\n{post_text}"},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=1000,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content or ""
            return _extract_json(raw)
        except (json.JSONDecodeError, AttributeError) as exc:
            if attempt == max_retries:
                print(f"  [WARN] JSON parse failed after {max_retries} attempts: {exc}")
                return {
                    "claims": [],
                    "verdicts": [],
                    "overall_label": "unverifiable",
                    "reasoning": f"Parse error: {exc}",
                }
            time.sleep(1)
    return {}  # unreachable


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> Dict[str, Any]:
    """Run one complete episode for the given task_id. Returns result dict."""

    # --- Reset ---
    resp = requests.post(f"{API_BASE_URL}/reset", params={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    session_id: str = data["session_id"]
    obs: Dict[str, Any] = data["observation"]

    print(f"\n{'─'*60}")
    print(f"[START]  task={obs['task_id']}  difficulty={obs['task_difficulty']}")
    print(f"[POST]   {obs['post_text'][:120]}{'...' if len(obs['post_text']) > 120 else ''}")

    episode_reward = 0.0
    done = False
    step = 0

    while not done and step < obs["max_steps"]:
        # --- LLM inference ---
        action_dict = _call_model(obs["post_text"])

        print(f"\n[STEP {step+1}]")
        print(f"  Claims   : {action_dict.get('claims', [])}")
        print(f"  Verdicts : {action_dict.get('verdicts', [])}")
        print(f"  Label    : {action_dict.get('overall_label', '')}")
        print(f"  Reasoning: {action_dict.get('reasoning', '')[:100]}")

        # --- Submit action ---
        step_resp = requests.post(
            f"{API_BASE_URL}/step/{session_id}",
            json=action_dict,
            timeout=30,
        )
        step_resp.raise_for_status()
        step_data = step_resp.json()

        reward_val: float = step_data["reward"]["value"]
        breakdown: Dict = step_data["reward"]["breakdown"]
        done = step_data["done"]
        obs = step_data["observation"]

        episode_reward += reward_val
        step += 1

        print(f"  Reward   : {reward_val:.4f}")
        print(f"  Breakdown:")
        for k, v in breakdown.items():
            print(f"    {k}: {v}")

    print(f"\n[END]  episode_reward={episode_reward:.4f}  steps={step}")
    return {
        "task_id": task_id,
        "episode_reward": round(episode_reward, 4),
        "steps": step,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASK_IDS = ["easy_brain_myth", "medium_mixed_facts", "hard_misleading_vaers"]


def main() -> None:
    print("=" * 60)
    print("MisinfoPatrolEnv — Baseline Inference")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print("=" * 60)

    results: List[Dict[str, Any]] = []

    for task_id in TASK_IDS:
        try:
            result = run_episode(task_id)
            results.append(result)
        except Exception as exc:
            print(f"[ERROR] Task {task_id} failed: {exc}")
            results.append({"task_id": task_id, "episode_reward": 0.0, "steps": 0, "error": str(exc)})

    # --- Summary ---
    print(f"\n{'='*60}")
    print("BASELINE SCORES")
    print(f"{'─'*60}")
    total = 0.0
    for r in results:
        score = r["episode_reward"]
        total += score
        status = r.get("error", "ok")
        print(f"  {r['task_id']:<30} {score:.4f}   [{status}]")
    avg = total / len(results) if results else 0.0
    print(f"{'─'*60}")
    print(f"  {'AVERAGE':<30} {avg:.4f}")
    print("=" * 60)

    # Exit non-zero if any task errored
    if any("error" in r for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
