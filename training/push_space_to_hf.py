from __future__ import annotations

"""
Upload the browser game to the Hugging Face Space repo.

Uses the HF API (same approach as the initial Space deploy). Files land at the
Space repo root (flat layout). Run from project root:

    python3 training/push_space_to_hf.py

Requires web/config.js with your hfUsername (copy from config.example.js).
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

from hf_config import hf_space_repo

SPACE_FILES = [
    "index.html",
    "game.js",
    "style.css",
    "config.js",
]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    web_root = root / "web"
    api = HfApi()
    space_repo = hf_space_repo()

    config_path = web_root / "config.js"
    if not config_path.is_file():
        print(
            "Missing web/config.js. Copy web/config.example.js and set hfUsername.",
            file=sys.stderr,
        )
        sys.exit(1)

    uploaded = 0
    for name in SPACE_FILES:
        local_path = web_root / name
        if not local_path.is_file():
            print(f"Skipping missing web/{name}")
            continue

        print(f"Uploading web/{name} -> {name}...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=name,
            repo_id=space_repo,
            repo_type="space",
            commit_message=f"Update {name}",
        )
        uploaded += 1

    if uploaded == 0:
        print("No files uploaded.", file=sys.stderr)
        sys.exit(1)

    username = space_repo.split("/", 1)[0]
    space_name = space_repo.split("/", 1)[1]
    print(f"Done. Space: https://huggingface.co/spaces/{space_repo}")
    print(f"Play: https://{username}-{space_name}.hf.space")


if __name__ == "__main__":
    main()
