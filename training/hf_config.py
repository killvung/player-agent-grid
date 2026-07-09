from __future__ import annotations

import os
import sys
from pathlib import Path


def load_dotenv() -> None:
    """Load project-root .env into os.environ (existing vars are not overwritten)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def hf_model_repo() -> str:
    load_dotenv()
    if repo := os.environ.get("HF_MODEL_REPO"):
        return repo

    username = os.environ.get("HF_USERNAME")
    model_name = os.environ.get("HF_MODEL_NAME", "player-agent-grid-policy")
    if not username:
        print(
            "Set HF_MODEL_REPO or HF_USERNAME (and optionally HF_MODEL_NAME). "
            "See .env.example.",
            file=sys.stderr,
        )
        sys.exit(1)

    return f"{username}/{model_name}"


def hf_space_repo() -> str:
    load_dotenv()
    if repo := os.environ.get("HF_SPACE_REPO"):
        return repo

    username = os.environ.get("HF_USERNAME")
    space_name = os.environ.get("HF_SPACE_NAME", "player-agent-grid-game")
    if not username:
        print(
            "Set HF_SPACE_REPO or HF_USERNAME (and optionally HF_SPACE_NAME). "
            "See .env.example.",
            file=sys.stderr,
        )
        sys.exit(1)

    return f"{username}/{space_name}"
