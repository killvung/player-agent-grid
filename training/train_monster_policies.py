from __future__ import annotations

"""
Train and export both monster policies:
- REINFORCE (online policy gradient)
- SARSA (tabular TD control)
"""

from train_monster_policy_reinforce import (
    export_policy as export_reinforce_policy,
    run_online_policy_training,
)
from train_monster_policy_sarsa import (
    export_policy as export_sarsa_policy,
    run_td_training,
)


def main() -> None:
    print("Training REINFORCE policy...")
    preferences = run_online_policy_training()
    export_reinforce_policy(preferences)
    print(
        f"Wrote trained_policies/monster_policy_reinforce.json "
        f"({len(preferences)} states)"
    )

    print("Training SARSA policy...")
    q_values = run_td_training()
    export_sarsa_policy(q_values)
    print(
        f"Wrote trained_policies/monster_policy_sarsa.json "
        f"({len(q_values)} states)"
    )

    print("Done.")


if __name__ == "__main__":
    main()
