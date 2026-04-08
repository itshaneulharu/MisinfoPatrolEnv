"""
MisinfoPatrolEnv — Core Environment
===================================
A real-world OpenEnv where an AI agent acts as a social media fact-checker.

The agent receives a viral post and must:
  1. Extract factual claims embedded in the post
  2. Assign verdicts to each claim (true / false / misleading / unverifiable)
  3. Label the post overall (credible / misinformation / misleading / unverifiable)

API: reset() → step() → state()   (standard OpenEnv interface)
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from graders import grade_action
from tasks import TASKS


# ---------------------------------------------------------------------------
# Typed Models (OpenEnv spec)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """State returned to the agent after reset() or step()."""
    post_id: str = Field(..., description="Unique identifier for the social-media post")
    post_text: str = Field(..., description="Full text of the viral social-media post to fact-check")
    task_id: str = Field(..., description="Identifier of the active task")
    task_difficulty: str = Field(..., description="Task difficulty: easy | medium | hard")
    instructions: str = Field(..., description="Agent instructions for this environment")
    step_count: int = Field(..., description="Number of steps taken in the current episode")
    max_steps: int = Field(..., description="Maximum steps allowed per episode")


class Action(BaseModel):
    """Action submitted by the agent."""
    claims: List[str] = Field(
        ...,
        description="Factual claims extracted from the post, one per list item",
    )
    verdicts: List[str] = Field(
        ...,
        description=(
            "Verdict for each claim in the same order as `claims`. "
            "Valid values: 'true', 'false', 'misleading', 'unverifiable'"
        ),
    )
    overall_label: str = Field(
        ...,
        description=(
            "Overall credibility label for the post. "
            "Valid values: 'credible', 'misinformation', 'misleading', 'unverifiable'"
        ),
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the fact-checking process and conclusion",
    )


class RewardInfo(BaseModel):
    """Structured reward returned after each step()."""
    value: float = Field(..., description="Composite reward in [0.0, 1.0]")
    breakdown: Dict[str, float] = Field(..., description="Per-component scores")
    done: bool = Field(..., description="Whether the episode has ended")
    info: Dict[str, Any] = Field(..., description="Auxiliary diagnostic information")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_INSTRUCTIONS = (
    "You are an AI fact-checker. Analyse the viral social-media post provided and:\n"
    "  1. Extract every distinct factual claim made in the post.\n"
    "  2. For each claim, assign a verdict: 'true', 'false', 'misleading', or 'unverifiable'.\n"
    "  3. Assign a single overall label to the post: "
    "'credible', 'misinformation', 'misleading', or 'unverifiable'.\n"
    "  4. Provide a concise reasoning string.\n"
    "Submit your response as an Action object."
)


class MisinfoPatrolEnv:
    """
    OpenEnv-compliant environment for AI-powered misinformation detection.

    Usage
    -----
    env = MisinfoPatrolEnv()
    obs = env.reset()
    action = Action(claims=[...], verdicts=[...], overall_label="...", reasoning="...")
    obs, reward, done, info = env.step(action)
    print(env.state())
    """

    metadata = {
        "name": "MisinfoPatrolEnv",
        "version": "1.0.0",
        "domain": "content_moderation",
        "tags": ["misinformation", "fact-checking", "nlp", "social-media"],
    }

    def __init__(
        self,
        task_id: Optional[str] = None,
        max_steps: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.task_id = task_id
        self.max_steps = max_steps
        self._rng = random.Random(seed)

        self._current_task: Optional[Dict[str, Any]] = None
        self._step_count: int = 0
        self._done: bool = False
        self._episode_reward: float = 0.0
        self._session_id: str = str(uuid.uuid4())
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Begin a new episode. Returns the initial observation."""
        self._step_count = 0
        self._done = False
        self._episode_reward = 0.0
        self._history = []
        self._session_id = str(uuid.uuid4())

        if self.task_id:
            task = next((t for t in TASKS if t["id"] == self.task_id), None)
            if task is None:
                raise ValueError(
                    f"Unknown task_id '{self.task_id}'. "
                    f"Valid IDs: {[t['id'] for t in TASKS]}"
                )
            self._current_task = task
        else:
            self._current_task = self._rng.choice(TASKS)

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, RewardInfo, bool, Dict[str, Any]]:
        """
        Apply an agent action and return (observation, reward, done, info).

        The agent should call reset() before the first step and after each
        episode ends (done=True).
        """
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")
        if self._current_task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        self._step_count += 1

        # Grade the action
        reward_value, breakdown = grade_action(action, self._current_task)
        self._episode_reward += reward_value

        # Episode ends when: max steps reached, or agent achieves near-perfect score
        done = self._step_count >= self.max_steps or reward_value >= 0.90
        self._done = done

        info: Dict[str, Any] = {
            "session_id": self._session_id,
            "step": self._step_count,
            "episode_reward": round(self._episode_reward, 4),
            "task_id": self._current_task["id"],
            "task_difficulty": self._current_task["difficulty"],
            "ground_truth_label": self._current_task["overall_label"],
        }

        reward_info = RewardInfo(
            value=reward_value,
            breakdown=breakdown,
            done=done,
            info=info,
        )

        # Log to history
        self._history.append(
            {
                "step": self._step_count,
                "action": action.model_dump(),
                "reward": reward_value,
                "breakdown": breakdown,
                "done": done,
            }
        )

        obs = self._make_observation()
        return obs, reward_info, done, info

    def state(self) -> Dict[str, Any]:
        """Return the full current state of the environment."""
        return {
            "session_id": self._session_id,
            "current_task": self._current_task,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "done": self._done,
            "episode_reward": round(self._episode_reward, 4),
            "history": self._history,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        assert self._current_task is not None
        return Observation(
            post_id=self._current_task["id"],
            post_text=self._current_task["post_text"],
            task_id=self._current_task["id"],
            task_difficulty=self._current_task["difficulty"],
            instructions=_INSTRUCTIONS,
            step_count=self._step_count,
            max_steps=self.max_steps,
        )
