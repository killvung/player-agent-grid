# Player-Agent Grid Learning Game

This project is a simple classroom game for learning player-agent interaction in a grid world.
The player tries to collect a key, unlock the door (goal), and survive while a monster agent learns/acts using selectable policies (online policy gradient, TD SARSA, or greedy baseline).

## What This Project Includes

- `web/`: browser game (player movement, monster behavior, debug panel, policy switcher)
- `training/`: Python training scripts for monster policies (Gymnasium env)
  - `env.py`: `MonsterGridEnv` Gymnasium environment
  - `train_monster_policy_reinforce.py`: online policy gradient (REINFORCE-style)
  - `train_monster_policy_sarsa.py`: TD SARSA control
- `trained_policies/`: generated policy artifacts (ignored by git)

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
python3 training/train_monster_policy_reinforce.py
```

Generate TD policy:

```bash
python3 training/train_monster_policy_sarsa.py
```

Outputs:

- `trained_policies/monster_policy_reinforce.json`
- `trained_policies/monster_policy_sarsa.json`

## Policy Modes in Game

- **Online policy gradient**: uses `monster_policy_reinforce.json` when state key exists.
- **TD SARSA**: uses `monster_policy_sarsa.json` when state key exists.
- **Greedy chase**: no policy file; moves to reduce distance to player.
- Missing state fallback: greedy chase.

## Deploy

This project uses three repositories:

| Repo | URL | What lives here |
|------|-----|-----------------|
| **GitHub** (source code) | [killvung/player-agent-grid](https://github.com/killvung/player-agent-grid) | Training scripts, web game, docs |
| **HF Space** (live game) | [player-agent-grid-game](https://huggingface.co/spaces/killvung/player-agent-grid-game) | Playable browser game |
| **HF Model** (policies) | [player-agent-grid-policy](https://huggingface.co/killvung/player-agent-grid-policy) | Trained policy JSON files |

The web game loads policies at runtime from the HF model repo (`web/game.js`). Policy files in `trained_policies/` are gitignored and are **not** pushed to GitHub.

### Push code to GitHub

```bash
git add -A
git commit -m "Describe your change"
git push origin main
```

Commit: `training/`, `web/`, `README.md`, `requirements.txt`, `.gitignore`  
Do not commit: `trained_policies/*.json`, `.venv/`

### Deploy the game (HF Space)

**Option A — link GitHub (recommended, one-time setup):**

1. Open [Space Settings → Repository](https://huggingface.co/spaces/killvung/player-agent-grid-game/settings)
2. Connect `killvung/player-agent-grid`, branch `main`
3. In Space settings, set **App file** to `web/index.html` (Static SDK)

After linking, `git push origin main` updates GitHub and rebuilds the Space automatically.

**Option B — upload manually with the HF API:**

```bash
source .venv/bin/activate
pip install huggingface_hub
huggingface-cli login

python <<'PY'
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
root = Path("web")
for name in ["index.html", "game.js", "style.css"]:
    api.upload_file(
        path_or_fileobj=(root / name).read_bytes(),
        path_in_repo=name,
        repo_id="killvung/player-agent-grid-game",
        repo_type="space",
        commit_message=f"Update {name}",
    )
PY
```

Use Option B when you want to update the Space without pushing to GitHub.

### Upload policies (HF Model repo)

After retraining locally:

```bash
python3 training/train_monster_policy_reinforce.py
python3 training/train_monster_policy_sarsa.py
```

Upload the new files (Space picks them up on next page load; no Space redeploy needed):

```bash
source .venv/bin/activate

python <<'PY'
from huggingface_hub import HfApi

api = HfApi()
repo = "killvung/player-agent-grid-policy"
for local, remote in [
    ("trained_policies/monster_policy_reinforce.json", "monster_policy_reinforce.json"),
    ("trained_policies/monster_policy_sarsa.json", "monster_policy_sarsa.json"),
]:
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=repo,
        repo_type="model",
        commit_message=f"Update {remote}",
    )
PY
```

### Typical workflow

```bash
# 1. Edit code or game UI → GitHub (and Space if linked)
git push origin main

# 2. Retrain policies → HF model repo only
python3 training/train_monster_policy_reinforce.py
python3 training/train_monster_policy_sarsa.py
# then run the upload script above
```

Play online: [player-agent-grid-game](https://huggingface.co/spaces/killvung/player-agent-grid-game)

## Debug / Teaching Features

- Toggle debug panel in UI.
- Adjustable epsilon slider for explore vs exploit behavior.
- Shows selected action source (`policy_table` vs fallback), episode counts, and recent decisions.
