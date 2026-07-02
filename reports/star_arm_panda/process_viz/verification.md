# Panda Gripper Process Verification

- audit_summary.json satisfies actor mean safe, high-risk corridor shadow, positive corridor lift, and no risky shadow execution.
- selected seed: `900078`
- selected audit steps: `0, 14, 16, 22`
- gripper/finger links are read from PyBullet link states 11/9/10.
- audit_sequence_rows.csv contains real critic queries from replayed states plus the logged selected audit snapshot for step 16.
