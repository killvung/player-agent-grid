from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Set, Tuple

Pos = Tuple[int, int]


@dataclass
class StepResult:
    next_state: str
    reward: float
    done: bool
    info: dict


class GridWorld:
    ACTIONS = ["up", "down", "left", "right"]
    DELTAS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    def __init__(self, rows: int = 8, cols: int = 8, seed: int = 0):
        self.rows = rows
        self.cols = cols
        self.rng = random.Random(seed)
        self.barriers: Set[Pos] = set()
        self.player: Pos = (0, 0)
        self.monster: Pos = (0, 0)
        self.goal: Pos = (0, 0)

    def reset(self, barrier_count: int = 8) -> str:
        self.barriers = self._random_barriers(barrier_count)
        self.player, self.monster, self.goal = self._random_entities()
        return self.state_key()

    def state_key(self) -> str:
        return f"p={self.player[0]},{self.player[1]}|m={self.monster[0]},{self.monster[1]}|g={self.goal[0]},{self.goal[1]}"

    def valid_moves(self, pos: Pos) -> List[str]:
        valid = []
        for action in self.ACTIONS:
            nxt = self._apply_move(pos, action)
            if nxt != pos:
                valid.append(action)
        return valid

    def step(self, monster_action: str) -> StepResult:
        if monster_action not in self.ACTIONS:
            monster_action = self.rng.choice(self.ACTIONS)

        self.monster = self._apply_move(self.monster, monster_action)

        if self.monster == self.player:
            return StepResult(self.state_key(), 1.0, True, {"winner": "monster"})
        if self.player == self.goal:
            return StepResult(self.state_key(), -1.0, True, {"winner": "player"})

        return StepResult(self.state_key(), 0.0, False, {})

    def set_player_action(self, action: str) -> None:
        self.player = self._apply_move(self.player, action)

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
        self.rng.shuffle(cells)
        return set(cells[: max(0, min(count, len(cells) - 3))])

    def _random_entities(self) -> Tuple[Pos, Pos, Pos]:
        free_cells = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.barriers]
        self.rng.shuffle(free_cells)
        player, monster, goal = free_cells[:3]
        return player, monster, goal
