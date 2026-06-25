#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diagnostics.shadow_oracle import snapshot_supported  # noqa: E402
from main_star import make_env, reset_env  # noqa: E402


PROBE_ATTRS = [
    "model",
    "data",
    "task",
    "agent",
    "engine",
    "mujoco",
    "_elapsed_steps",
    "elapsed_steps",
    "spec",
    "np_random",
]


def scalar(value: Any):
    if isinstance(value, (str, int, float, bool, type(None), np.integer, np.floating, np.bool_)):
        return value.item() if hasattr(value, "item") else value
    return None


def describe_obj(value: Any) -> dict:
    if value is None:
        return {"exists": False}
    out = {
        "exists": True,
        "type": f"{type(value).__module__}.{type(value).__name__}",
    }
    simple = scalar(value)
    if simple is not None:
        out["value"] = simple
    for attr in ("qpos", "qvel", "time", "ctrl", "act", "qacc_warmstart", "mocap_pos", "mocap_quat"):
        if hasattr(value, attr):
            item = getattr(value, attr)
            shape = getattr(item, "shape", None)
            out[f"{attr}_shape"] = list(shape) if shape is not None else None
    return out


def wrapper_chain(env) -> list[Any]:
    chain = []
    current = env
    for _ in range(80):
        chain.append(current)
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and all(unwrapped is item for item in chain) is False and unwrapped not in chain:
        chain.append(unwrapped)
    return chain


def inspect_env(task: str, *, safe_env: bool, binary_cost: bool, seed: int) -> dict:
    env = make_env(task, safe_env=safe_env, train=False, binary_cost=binary_cost)
    try:
        obs = reset_env(env, seed=seed)
        supported, reason = snapshot_supported(env)
        rows = []
        for index, wrapper in enumerate(wrapper_chain(env)):
            row = {
                "index": index,
                "type": f"{type(wrapper).__module__}.{type(wrapper).__name__}",
                "repr": repr(wrapper)[:240],
            }
            for attr in PROBE_ATTRS:
                if hasattr(wrapper, attr):
                    try:
                        row[attr] = describe_obj(getattr(wrapper, attr))
                    except Exception as exc:
                        row[attr] = {"exists": True, "error": f"{type(exc).__name__}: {exc}"}
            # Search common nested paths without assuming Safety-Gymnasium internals.
            for prefix in ("task", "engine", "agent"):
                if hasattr(wrapper, prefix):
                    obj = getattr(wrapper, prefix)
                    for attr in ("model", "data"):
                        key = f"{prefix}.{attr}"
                        if hasattr(obj, attr):
                            row[key] = describe_obj(getattr(obj, attr))
            rows.append(row)
        return {
            "task": task,
            "safe_env": safe_env,
            "binary_cost": binary_cost,
            "seed": seed,
            "reset_obs_shape": list(np.asarray(obs).shape),
            "snapshot_supported": supported,
            "snapshot_reason": reason,
            "wrapper_chain": rows,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="SafetyPointGoal1-v0,SafetyCarGoal1-v0")
    parser.add_argument("--safe-env", default="True")
    parser.add_argument("--binary-cost", default="True")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="reports/star_v2_final/oracle")
    args = parser.parse_args()
    safe_env = str(args.safe_env).lower() == "true"
    binary_cost = str(args.binary_cost).lower() == "true"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for task in [item.strip() for item in args.tasks.split(",") if item.strip()]:
        result = inspect_env(task, safe_env=safe_env, binary_cost=binary_cost, seed=args.seed)
        path = output / f"inspect_{task.replace('-', '_')}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True))
        summaries.append(
            {
                "task": task,
                "snapshot_supported": result["snapshot_supported"],
                "snapshot_reason": result["snapshot_reason"],
                "path": str(path),
            }
        )
        print(f"{task}: supported={result['snapshot_supported']} reason={result['snapshot_reason']} path={path}")
    (output / "inspect_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
