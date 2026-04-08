"""
MisinfoPatrolEnv — Graders
Programmatic scoring with partial credit across three dimensions:
  1. Claim Extraction    (0.0–1.0, weight 0.30)
  2. Verdict Accuracy    (0.0–1.0, weight 0.40)
  3. Overall Label       (0.0–1.0, weight 0.30)

Rewards partial progress throughout the trajectory, not just at terminal states.
"""

import re
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.strip().lower()


def _keyword_overlap(a: str, b: str) -> float:
    """Jaccard-style overlap on meaningful words (≥4 chars)."""
    kw_a = set(re.findall(r"\b\w{4,}\b", _normalize(a)))
    kw_b = set(re.findall(r"\b\w{4,}\b", _normalize(b)))
    if not kw_a and not kw_b:
        return 0.0
    return len(kw_a & kw_b) / max(len(kw_a | kw_b), 1)


def _match_claims(
    agent_claims: List[str],
    true_claims: List[str],
    threshold: float = 0.30,
) -> List[Tuple[int, int, float]]:
    """
    Greedily match each true claim to the best agent claim.
    Returns list of (true_idx, agent_idx, overlap_score).
    """
    matches: List[Tuple[int, int, float]] = []
    used_agent: set = set()

    for ti, tc in enumerate(true_claims):
        best_score, best_ai = 0.0, -1
        for ai, ac in enumerate(agent_claims):
            if ai in used_agent:
                continue
            score = _keyword_overlap(tc, ac)
            if score > best_score:
                best_score, best_ai = score, ai
        if best_score >= threshold and best_ai != -1:
            matches.append((ti, best_ai, best_score))
            used_agent.add(best_ai)

    return matches


# ---------------------------------------------------------------------------
# Sub-graders
# ---------------------------------------------------------------------------

VALID_VERDICTS = {"true", "false", "misleading", "unverifiable"}

# Pairs where partial credit is awarded (both in the "problematic" family)
PARTIAL_CREDIT_PAIRS = {
    frozenset({"false", "misleading"}),
    frozenset({"false", "unverifiable"}),
    frozenset({"misleading", "unverifiable"}),
}


def _grade_claim_extraction(agent_claims: List[str], true_claims: List[str]) -> float:
    """Score how completely the agent identified the key claims. [0.0–1.0]"""
    if not agent_claims:
        return 0.0
    if not true_claims:
        return 0.0
    matches = _match_claims(agent_claims, true_claims)
    # Weighted by match quality
    weighted = sum(score for _, _, score in matches)
    return min(weighted / len(true_claims), 1.0)


def _grade_verdicts(
    agent_claims: List[str],
    agent_verdicts: List[str],
    true_claims: List[str],
    true_verdicts: List[str],
) -> float:
    """Score verdict accuracy for matched claims. [0.0–1.0]"""
    if not agent_claims:
        return 0.0
    matches = _match_claims(agent_claims, true_claims)
    if not matches:
        return 0.0

    total_score = 0.0
    for ti, ai, _ in matches:
        if ai >= len(agent_verdicts):
            continue
        av = _normalize(agent_verdicts[ai])
        tv = _normalize(true_verdicts[ti])

        if av not in VALID_VERDICTS:
            continue

        if av == tv:
            total_score += 1.0
        elif frozenset({av, tv}) in PARTIAL_CREDIT_PAIRS:
            total_score += 0.5  # partial credit for wrong-direction-but-right-sign

    return total_score / len(true_claims)


def _grade_overall_label(agent_label: str, true_label: str) -> float:
    """Score the overall post label. [0.0–1.0]"""
    VALID_LABELS = {"credible", "misinformation", "misleading", "unverifiable"}
    al = _normalize(agent_label)
    tl = _normalize(true_label)

    if al == tl:
        return 1.0
    # Partial credit: misleading ↔ misinformation (both capture 'bad')
    if frozenset({al, tl}) == frozenset({"misleading", "misinformation"}):
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Main grader
# ---------------------------------------------------------------------------

def grade_action(action: Any, task: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Grade a single agent action against task ground truth.

    Returns:
        total_reward (float): composite score in [0.0, 1.0]
        breakdown (dict):     per-component weighted scores
    """
    # --- Claim extraction (30 %) ---
    claim_raw = _grade_claim_extraction(action.claims, task["claims"])
    claim_weighted = round(claim_raw * 0.30, 4)

    # --- Verdict accuracy (40 %) ---
    verdict_raw = _grade_verdicts(
        action.claims, action.verdicts, task["claims"], task["claim_verdicts"]
    )
    verdict_weighted = round(verdict_raw * 0.40, 4)

    # --- Overall label (30 %) ---
    label_raw = _grade_overall_label(action.overall_label, task["overall_label"])
    label_weighted = round(label_raw * 0.30, 4)

    # --- Penalty: penalise clearly undesirable behaviour ---
    # (e.g. submitting 0 claims or random garbage verdicts)
    penalty = 0.0
    if not action.claims:
        penalty += 0.10
    if not action.reasoning or len(action.reasoning) < 10:
        penalty += 0.05

    total = max(0.0, round(claim_weighted + verdict_weighted + label_weighted - penalty, 4))

    breakdown = {
        "claim_extraction_raw": round(claim_raw, 4),
        "claim_extraction_weighted": claim_weighted,
        "verdict_accuracy_raw": round(verdict_raw, 4),
        "verdict_accuracy_weighted": verdict_weighted,
        "overall_label_raw": round(label_raw, 4),
        "overall_label_weighted": label_weighted,
        "penalty": round(penalty, 4),
        "total": total,
    }

    return total, breakdown
