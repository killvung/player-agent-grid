from __future__ import annotations

from typing import List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Pos = Tuple[int, int]


class MonsterGridEnv(gym.Env):
    """
    Gymnasium environment for monster-vs-player grid chase.

    The agent controls the monster. The player may move stochastically during
  training to approximate a human opponent in the browser game.
    """

    metadata = {"render_modes": []}
    ACTION_NAMES = ["up", "down", "left", "right"]
    DELTAS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        barrier_count: int = 24,
        player_move_prob: float = 0.0,
        max_steps: int = 100,
    ):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.barrier_count = barrier_count
        self.player_move_prob = player_move_prob
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)
        # [player_r, player_c, monster_r, monster_c, goal_r, goal_c]
        self.observation_space = spaces.Box(
            low=0,
            high=max(rows, cols) - 1,
            shape=(6,),
            dtype=np.int32,
        )

        self.barriers: Set[Pos] = set()
        self.player: Pos = (0, 0)
        self.monster: Pos = (0, 0)
        self.goal: Pos = (0, 0)
        self._step_count = 0

    def state_key(self) -> str:
        return (
            f"p={self.player[0]},{self.player[1]}|"
            f"m={self.monster[0]},{self.monster[1]}|"
            f"g={self.goal[0]},{self.goal[1]}"
        )

    def valid_moves(self, pos: Pos) -> List[str]:
        valid = []
        for action in self.ACTION_NAMES:
            nxt = self._apply_move(pos, action)
            if nxt != pos:
                valid.append(action)
        return valid

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = options or {}
        barrier_count = int(options.get("barrier_count", self.barrier_count))

        self.barriers = self._random_barriers(barrier_count)
        self.player, self.monster, self.goal = self._random_entities()
        self._step_count = 0

        info = {"state_key": self.state_key()}
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            action = int(self.np_random.integers(0, 4))

        if self.player_move_prob > 0.0 and self.np_random.random() < self.player_move_prob:
            moves = self.valid_moves(self.player)
            if moves:
                self.player = self._apply_move(self.player, self.np_random.choice(moves))

        monster_action = self.ACTION_NAMES[action]
        self.monster = self._apply_move(self.monster, monster_action)
        self._step_count += 1

        terminated = False
        reward = 0.0
        winner = None

        if self.monster == self.player:
            terminated = True
            reward = 1.0
            winner = "monster"
        elif self.player == self.goal:
            terminated = True
            reward = -1.0
            winner = "player"

        truncated = self._step_count >= self.max_steps and not terminated
        info = {"state_key": self.state_key(), "winner": winner}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.player[0],
                self.player[1],
                self.monster[0],
                self.monster[1],
                self.goal[0],
                self.goal[1],
            ],
            dtype=np.int32,
        )

    def _apply_move(self, pos: Pos, action: str) -> Pos:
        dr, dc = self.DELTAS[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        candidate = (nr, nc)
        if not self._in_bounds(candidate):
            return pos
        if candidate in self.barriers:
            return pos
        return candidate

    def _in_bounds(self, pos: Pos) -> bool:
        return 0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols

    def _random_barriers(self, count: int) -> Set[Pos]:
        cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.np_random.shuffle(cells)
        return set(cells[: max(0, min(count, len(cells) - 3))])

    def _random_entities(self) -> Tuple[Pos, Pos, Pos]:
        free_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.barriers
        ]
        self.np_random.shuffle(free_cells)
        player, monster, goal = free_cells[:3]
        return player, monster, goal


# Backward-compatible alias used in older scripts/docs.
GridWorld = MonsterGridEnv
