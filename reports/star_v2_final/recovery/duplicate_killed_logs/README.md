# Duplicate killed logs

These logs came from an accidental manual refill at 2026-06-27 around 04:21 that relaunched already-completed PointGoal1 pointwise resume_300k sessions. The duplicate sessions were immediately stopped to protect completed final checkpoints. They are archived here so the main resume_300k log scan does not treat the intentional duplicate-session KeyboardInterrupt as a real experiment failure.
