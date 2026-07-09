from __future__ import annotations

import os
import sys


def hf_model_repo() -> str:
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
