🧠 MisinfoPatrolEnv

OpenEnv · Content Moderation · Fact-Checking · NLP

MisinfoPatrolEnv is a real-world reinforcement learning environment where an AI agent learns to detect and classify misinformation in viral social media posts. The agent extracts factual claims, evaluates their validity, and assigns an overall credibility label using structured reasoning and reward feedback.

🚀 Motivation

Misinformation detection is one of the most critical challenges in modern NLP.
Most systems treat it as a single-label classification problem, ignoring reasoning.

MisinfoPatrolEnv changes this by modeling fact-checking as a sequential decision process, where:

Claims must be extracted
Each claim must be evaluated
Final judgment must be justified

This mirrors how real human fact-checkers work.

🏗️ Environment Overview
Property	Value
Domain	Content moderation / fact-checking
Interface	OpenEnv (step / reset / state)
Observation	Viral post + metadata
Action	Claims + verdicts + label + reasoning
Reward	Dense composite [0.0–1.0]
Max steps	3
Tasks	3 (easy → hard)
🔍 Observation Space
class Observation(BaseModel):
    post_id: str
    post_text: str
    task_id: str
    task_difficulty: str
    instructions: str
    step_count: int
    max_steps: int
🎯 Action Space
class Action(BaseModel):
    claims: List[str]
    verdicts: List[str]
    overall_label: str
    reasoning: str
🧩 Tasks
🟢 Task 1 — easy_brain_myth

Classic myth disguised as breaking news.

👉 Detect misinformation + reject emotional framing
Expected: misinformation

🟡 Task 2 — medium_mixed_facts

Mix of true and false claims.

👉 Requires claim-level reasoning
Expected: misinformation

🔴 Task 3 — hard_misleading_vaers

Technically true data used misleadingly.

👉 Requires contextual understanding
Expected: misleading

⚖️ Reward Function
Component	Weight
Claim extraction	30%
Verdict accuracy	40%
Overall label	30%
Penalties
Empty claims → −0.10
Missing reasoning → −0.05
Termination
Max steps reached OR
Reward ≥ 0.90
🧪 API Usage
Start Episode
import requests

BASE = "http://localhost:7860"

res = requests.post(f"{BASE}/reset")
data = res.json()

session_id = data["session_id"]
obs = data["observation"]
Take Step
action = {
    "claims": ["humans only use 10% of their brains"],
    "verdicts": ["false"],
    "overall_label": "misinformation",
    "reasoning": "This myth is scientifically debunked."
}

res = requests.post(f"{BASE}/step/{session_id}", json=action)
print(res.json())
🧠 Python Usage
from environment import MisinfoPatrolEnv, Action

env = MisinfoPatrolEnv()
obs = env.reset()

action = Action(
    claims=["humans only use 10% of brain"],
    verdicts=["false"],
    overall_label="misinformation",
    reasoning="Neuroscience disproves this claim."
)

obs, reward, done, info = env.step(action)
print(reward.value)
🤖 Baseline Inference
export API_BASE_URL=https://YOUR_SPACE_URL
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_your_token

python inference.py
🐳 Docker
docker build -t misinfo-patrol-env .
docker run -p 7860:7860 misinfo-patrol-env
💻 Local Development
pip install -r requirements.txt
uvicorn app:app --port 7860
📁 Project Structure
MisinfoPatrolEnv/
├── app.py
├── environment.py
├── graders.py
├── tasks.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
✅ Features
OpenEnv-compliant API
Dense reward shaping
Multi-step reasoning
Real-world misinformation scenarios
Baseline LLM agent
📜 License

MIT © 2026 — MisinfoPatrolEnv Team