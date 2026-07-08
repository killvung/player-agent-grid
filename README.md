# Player-Agent Grid Learning Game

This project is a simple classroom game for learning player-agent interaction in a grid world.
The player tries to collect a key, unlock the door (goal), and survive while a monster agent learns/acts using selectable policies (online policy gradient, TD SARSA, or greedy baseline).

## What This Project Includes

- `web/`: browser game (player movement, monster behavior, debug panel, policy switcher)
- `training/`: Python training scripts for monster policies (Gymnasium env)
  - `env.py`: `MonsterGridEnv` Gymnasium environment
  - `train_policy_online.py`: online policy gradient (REINFORCE-style)
  - `train_policy_td.py`: TD SARSA control
- `data/`: generated policy artifacts (ignored by git)

## Game Rules

- Player (blue) moves with arrow keys or WASD.
- Monster (red) moves on its own timer (not tied to player input).
- Player must pick up the key (yellow) before entering the goal door.
- Door/goal is orange when locked, green when unlocked.
- If monster reaches player: monster wins.
- If player reaches unlocked goal: player wins.

## Run Locally

From project root:

```bash
python3 -m http.server 8000
```

Open:

- `http://localhost:8000/web/index.html`

## Train Policies

Install Python dependencies (Gymnasium):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate online policy:

```bash
python3 training/train_policy_online.py
```

Generate TD policy:

```bash
python3 training/train_policy_td.py
```

Outputs:

- `data/policy.json`
- `data/policy_td.json`

## Policy Modes in Game

- **Online policy gradient**: uses `policy.json` when state key exists.
- **TD SARSA**: uses `policy_td.json` when state key exists.
- **Greedy chase**: no policy file; moves to reduce distance to player.
- Missing state fallback: greedy chase.

## Debug / Teaching Features

- Toggle debug panel in UI.
- Adjustable epsilon slider for explore vs exploit behavior.
- Shows selected action source (`policy_table` vs fallback), episode counts, and recent decisions.

## Model Repository

Hugging Face policy model:

- [player-agent-grid-policy](https://huggingface.co/killvung/player-agent-grid-policy/tree/main)
