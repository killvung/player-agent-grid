from __future__ import annotations

"""
TD policy training (tabular SARSA(0) control).

What this trains:
- A monster action-value table Q(s,a) over 4 discrete actions.
- The exported policy is greedy w.r.t. Q: π(s) = argmax_a Q(s,a).

Key idea:
- SARSA is a Temporal-Difference (TD) method: it bootstraps from the next state's value.
- Update uses the TD error:
    δ = r + γ Q(s',a') - Q(s,a)
  and then Q(s,a) ← Q(s,a) + α δ

Notes for this specific classroom game:
- The environment currently has a moving player in the browser.
  During training we approximate that by randomly moving the player sometimes
  (see maybe_move_player). This keeps the monster from overfitting to a static player.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from env import GridWorld


def argmax(values: List[float]) -> int:
    best_idx = 0
    best_val = values[0]
    for i, v in enumerate(values):
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


def epsilon_greedy(values: List[float], epsilon: float, rng: random.Random) -> int:
    """Choose a random action with prob ε, otherwise greedy action."""
    if rng.random() < epsilon:
        return rng.randrange(len(values))
    return argmax(values)


def maybe_move_player(env: GridWorld, rng: random.Random, move_prob: float = 0.7) -> None:
    """
    Simple training-time approximation of an interactive opponent.

    In the actual web game, the player is controlled by a human.
    For offline training we inject stochastic player motion so the monster learns
    to respond to changing player positions rather than a frozen target.
    """
    if rng.random() > move_prob:
        return
    moves = env.valid_moves(env.player)
    if moves:
        env.set_player_action(rng.choice(moves))


def run_td_training(
    episodes: int = 8000,
    max_steps: int = 100,
    alpha: float = 0.12,
    gamma: float = 0.95,
    epsilon_start: float = 0.9,
    epsilon_end: float = 0.05,
) -> Dict[str, List[float]]:
    env = GridWorld(rows=8, cols=8, seed=7)
    rng = random.Random(7)

    # q_values[s][a] is the learned expected discounted return if the monster takes action a in state s.
    q_values: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    for ep in range(episodes):
        state = env.reset(barrier_count=8)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / max(1, episodes - 1))

        # SARSA is "on-policy": we select an action with the current epsilon-greedy policy
        # and we bootstrap from the value of the *next action chosen by that same policy*.
        action_idx = epsilon_greedy(q_values[state], epsilon, rng)
        for _ in range(max_steps):
            maybe_move_player(env, rng)

            action = env.ACTIONS[action_idx]
            result = env.step(action)
            next_state = result.next_state

            if result.done:
                # Terminal transition: no bootstrap term.
                td_target = result.reward
                td_error = td_target - q_values[state][action_idx]
                q_values[state][action_idx] += alpha * td_error
                break

            # Choose next action according to the current behavior policy (epsilon-greedy).
            next_action_idx = epsilon_greedy(q_values[next_state], epsilon, rng)

            # TD target includes bootstrap estimate of Q(s',a').
            td_target = result.reward + gamma * q_values[next_state][next_action_idx]
            td_error = td_target - q_values[state][action_idx]
            q_values[state][action_idx] += alpha * td_error

            state = next_state
            action_idx = next_action_idx

    return dict(q_values)


def export_policy(q_values: Dict[str, List[float]], out_path: str | None = None) -> None:
    # Export a deterministic greedy policy table for the browser: state_key -> argmax action.
    actions = ["up", "down", "left", "right"]
    policy = {state: actions[argmax(values)] for state, values in q_values.items()}
    example_state, example_action = ("p=1,2|m=1,1|g=6,5", "right")
    if policy:
        example_state, example_action = next(iter(policy.items()))

    payload = {
        "meta": {
            "algo": "td_sarsa",
            "actions": actions,
            "notes": "Tabular TD control using SARSA(0) with epsilon-greedy exploration.",
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
        out_path = str(root / "data" / "policy_td.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    q = run_td_training()
    export_policy(q)
    print("Wrote TD policy JSON to data/policy_td.json")
