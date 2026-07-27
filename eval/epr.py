"""Effective Prompt Ratio -- the project's signature metric (v2 §2.2).

GRPO advantage is group-normalized: A_i = (r_i - mean(r)) / std(r). When every
rollout in a group has the SAME reward, std=0 => advantage=0 => the prompt
contributes zero gradient.

EPR(t) = fraction of prompts in a training batch whose reward group has nonzero
variance (i.e. actually produces gradient). Logged for the whole run, per reward
type (binary vs partial) and stratified by difficulty (easy/medium/hard).

Expected story: EPR_binary starts low (hard prompts all-fail) and decays (easy
prompts saturate); EPR_partial is higher and decays slower. This is the core blog
figure and the empirical answer to "why does partial credit help".
"""

from __future__ import annotations


def group_has_gradient(rewards: list[float], tol: float = 1e-8) -> bool:
    """True iff the reward group has nonzero variance (nonzero advantage)."""
    if not rewards:
        return False
    return (max(rewards) - min(rewards)) > tol


def batch_epr(group_rewards: list[list[float]]) -> float:
    """EPR for one batch: fraction of prompt-groups with nonzero variance."""
    if not group_rewards:
        return 0.0
    n = sum(1 for g in group_rewards if group_has_gradient(g))
    return n / len(group_rewards)


def stratified_epr(group_rewards_by_difficulty: dict[str, list[list[float]]]) -> dict[str, float]:
    """EPR computed separately per difficulty layer (easy/medium/hard)."""
    return {d: batch_epr(groups) for d, groups in group_rewards_by_difficulty.items()}


class EPRLogger:
    """Accumulates EPR(t) curves for wandb / analysis.

    TODO: hook into the GRPO training step; record step, reward_type, overall EPR,
    and per-difficulty EPR each optimization step. Persist to results/ for the
    analysis notebook.
    """

    def __init__(self, reward_type: str):
        self.reward_type = reward_type
        self.history: list[dict] = []

    def log_step(self, step: int, group_rewards_by_difficulty: dict[str, list[list[float]]]) -> None:
        raise NotImplementedError
