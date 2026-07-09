from __future__ import annotations

"""
Upload trained policy JSON files to the Hugging Face model repo.

The web game and Space fetch policies from this repo at runtime.
Run after training:

    python3 training/train_monster_policies.py
    python3 training/push_policies_to_hf.py
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

from hf_config import hf_model_repo

POLICY_FILES = [
    ("trained_policies/monster_policy_reinforce.json", "monster_policy_reinforce.json"),
    ("trained_policies/monster_policy_sarsa.json", "monster_policy_sarsa.json"),
]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    api = HfApi()
    model_repo = hf_model_repo()

    for local_rel, remote_name in POLICY_FILES:
        local_path = root / local_rel
        if not local_path.is_file():
            print(f"Missing {local_rel}. Train policies first.", file=sys.stderr)
            sys.exit(1)

        print(f"Uploading {local_rel} -> {remote_name}...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=model_repo,
            repo_type="model",
            commit_message=f"Update {remote_name}",
        )

    print(f"Done. Policies live at https://huggingface.co/{model_repo}")
    print("Hard-refresh the Space to load the new files.")


if __name__ == "__main__":
    main()
