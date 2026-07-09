# Player-Agent Grid Learning Game

This project is a simple classroom game for learning player-agent interaction in a grid world.
The player tries to collect a key, unlock the door (goal), and survive while a monster agent learns/acts using selectable policies (online policy gradient, TD SARSA, or greedy baseline).

## What This Project Includes

- `web/`: browser game (player movement, monster behavior, debug panel, policy switcher)
- `training/`: Python training scripts for monster policies (Gymnasium env)
  - `env.py`: `MonsterGridEnv` Gymnasium environment
  - `train_monster_policies.py`: train both policies
  - `train_monster_policy_reinforce.py`: online policy gradient (REINFORCE-style)
  - `train_monster_policy_sarsa.py`: TD SARSA control
  - `push_policies_to_hf.py`: upload policies to Hugging Face
  - `push_space_to_hf.py`: upload web game to Hugging Face Space
- `trained_policies/`: generated policy artifacts (ignored by git)
- `.env.example` / `web/config.example.js`: Hugging Face username and repo config templates

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

For HF-hosted policies locally, copy `web/config.example.js` to `web/config.js` and set your username. Without `config.js`, the game loads from `trained_policies/` instead.

## Configuration

Copy the example env file and set your Hugging Face username:

```bash
cp .env.example .env
# edit .env: HF_USERNAME=your-hf-username
```

| Variable | Purpose |
|----------|---------|
| `HF_USERNAME` | Your Hugging Face account name |
| `HF_MODEL_NAME` | Model repo name (default: `player-agent-grid-policy`) |
| `HF_SPACE_NAME` | Space repo name (default: `player-agent-grid-game`) |
| `HF_MODEL_REPO` | Optional full model repo id (`username/model-name`) |
| `HF_SPACE_REPO` | Optional full Space repo id (`username/space-name`) |

For the web game, copy `web/config.example.js` to `web/config.js` (gitignored) with the same username.

## Train Policies

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # gymnasium, numpy, huggingface_hub
```

Generate both policies:

```bash
python3 training/train_monster_policies.py
```

Or individually:

```bash
python3 training/train_monster_policy_reinforce.py
```

```bash
python3 training/train_monster_policy_sarsa.py
```

Outputs:

- `trained_policies/monster_policy_reinforce.json`
- `trained_policies/monster_policy_sarsa.json`

Push policies to Hugging Face (for the Space and local web game):

```bash
python3 training/push_policies_to_hf.py
```

Loads `HF_USERNAME` from `.env` automatically. Requires `huggingface-cli login` once.

## Policy Modes in Game

- **Online policy gradient**: uses `monster_policy_reinforce.json` when state key exists.
- **TD SARSA**: uses `monster_policy_sarsa.json` when state key exists.
- **Greedy chase**: no policy file; moves to reduce distance to player.
- Missing state fallback: greedy chase.

## Deploy

This project uses three repositories:

| Repo | What lives here |
|------|-----------------|
| **GitHub** (source code) | Training scripts, web game, docs |
| **HF Space** (live game) | Playable browser game (`$HF_USERNAME/$HF_SPACE_NAME`) |
| **HF Model** (policies) | Trained policy JSON files (`$HF_USERNAME/$HF_MODEL_NAME`) |

The web game loads policies at runtime from your HF model repo (`web/config.js`). Policy files in `trained_policies/` are gitignored and are **not** pushed to GitHub.

### Push code to GitHub

```bash
git add -A
git commit -m "Describe your change"
git push origin main
```

Commit: `training/`, `web/`, `README.md`, `requirements.txt`, `.gitignore`, `.env.example`, `web/config.example.js`  
Do not commit: `trained_policies/*.json`, `.env`, `web/config.js`, `.venv/`

### Deploy the game (HF Space)

**Option A — link GitHub (recommended, one-time setup):**

1. Open your Space **Settings → Repository** on Hugging Face
2. Connect your GitHub repo, branch `main`
3. Set **App file** to `web/index.html` (Static SDK)
4. Add `web/config.js` to the Space separately (it is gitignored, so GitHub sync will not include it). Copy `web/config.example.js`, set `hfUsername`, and upload via the Space file editor or Option B below.

After linking, `git push origin main` updates GitHub and rebuilds the Space automatically.

**Option B — upload manually with the HF API** (flat Space layout at repo root):

```bash
cp web/config.example.js web/config.js   # set hfUsername first
python3 training/push_space_to_hf.py
```

Use Option B for a standalone Space repo, or to upload `config.js` without pushing to GitHub.

### Typical workflow

```bash
# 0. One-time setup
cp .env.example .env                  # set HF_USERNAME
cp web/config.example.js web/config.js  # set hfUsername
huggingface-cli login

# 1. Edit code or game UI → GitHub (and Space if linked)
git push origin main

# 2. Retrain policies → HF model repo
python3 training/train_monster_policies.py
python3 training/push_policies_to_hf.py
```

## Debug / Teaching Features

- Toggle debug panel in UI.
- Adjustable epsilon slider for explore vs exploit behavior.
- Shows selected action source (`policy_table` vs fallback), episode counts, and recent decisions.
