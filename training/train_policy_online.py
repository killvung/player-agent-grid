from __future__ import annotations

"""
Online policy training (tabular REINFORCE / policy gradient).

What this trains:
- A monster policy π(a|s) over 4 discrete actions: up/down/left/right.
- The policy is represented *tabularly* as per-state action preferences (logits).
- In each episode we sample actions from the current policy, observe rewards, then
  update preferences to increase the probability of actions that led to higher return.

Key idea:
- REINFORCE uses Monte Carlo returns: it does NOT bootstrap from value estimates.
- Update direction is proportional to: return * ∇ log π(a|s).
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from env import GridWorld


def argmax(values: List[float]) -> int:
    best_idx = 0
    best_val = values[0]
    for i, v in enumerate(values):
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


def softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    """Convert action-preferences (logits) into a probability distribution."""
    t = max(temperature, 1e-6)
    scaled = [v / t for v in logits]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    s = sum(exps)
    return [e / s for e in exps]


def sample_action(probs: List[float], rng: random.Random) -> int:
    """Sample an action index from a categorical distribution."""
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


def run_online_policy_training(
    episodes: int = 6000,
    max_steps: int = 100,
    gamma: float = 0.95,
    lr: float = 0.03,
    temperature: float = 1.0,
) -> Dict[str, List[float]]:
    env = GridWorld(rows=8, cols=8, seed=42)
    rng = random.Random(42)

    # preferences[s][a] are unnormalized action preferences (logits).
    # softmax(preferences[s]) gives π(a|s).
    preferences: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    for _ in range(episodes):
        state = env.reset(barrier_count=8)

        # Trajectory stores (state_key, chosen_action_index, immediate_reward).
        # We’ll compute discounted returns G_t from these rewards after the episode.
        trajectory: List[Tuple[str, int, float]] = []

        for _ in range(max_steps):
            probs = softmax(preferences[state], temperature=temperature)
            action_idx = sample_action(probs, rng)
            action = env.ACTIONS[action_idx]
            result = env.step(action)
            trajectory.append((state, action_idx, result.reward))

            state = result.next_state
            if result.done:
                break

        # Monte Carlo return:
        # G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ...
        returns: List[float] = []
        g = 0.0
        for _, _, reward in reversed(trajectory):
            g = reward + gamma * g
            returns.append(g)
        returns.reverse()

        # REINFORCE update:
        # For the taken action a_t, increase log-probability if G_t is positive,
        # decrease if G_t is negative.
        #
        # grad_log_pi for a softmax policy is:
        #   ∂ log π(a_t|s) / ∂ pref[i] = 1[i==a_t] - π(i|s)
        for (s_key, a_idx, _), g_t in zip(trajectory, returns):
            probs = softmax(preferences[s_key], temperature=temperature)
            for i in range(len(probs)):
                grad_log_pi = (1.0 if i == a_idx else 0.0) - probs[i]
                preferences[s_key][i] += lr * g_t * grad_log_pi

    return dict(preferences)


def export_policy(preferences: Dict[str, List[float]], out_path: str | None = None) -> None:
    # For browser use we export a *deterministic* policy table: state_key -> argmax action.
    # (At runtime the game can still add exploration via epsilon if you want.)
    actions = ["up", "down", "left", "right"]
    policy = {state: actions[argmax(values)] for state, values in preferences.items()}
    example_state, example_action = ("p=1,2|m=1,1|g=6,5", "right")
    if policy:
        example_state, example_action = next(iter(policy.items()))

    payload = {
        "meta": {
            "algo": "online_policy_gradient",
            "actions": actions,
            "notes": "Tabular softmax policy trained with online REINFORCE updates.",
            "how_to_read_policy": [
                "Each key in `policy` is a full game state string.",
                "State format is `p=row,col|m=row,col|g=row,col`.",
                "Value is the monster action for that exact state: one of up/down/left/right.",
                "If a state is missing in `policy`, the web game falls back to greedy chase logic.",
            ],
            "example": {
                "state_key": example_state,
                "chosen_action": example_action,
                "interpretation": "Given this player/monster/goal layout, monster should move in `chosen_action` direction.",
            },
        },
        "policy": policy,
    }

    if out_path is None:
        root = Path(__file__).resolve().parent.parent
        out_path = str(root / "data" / "policy.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    prefs = run_online_policy_training()
    export_policy(prefs)
    print("Wrote policy JSON to data/policy.json")
