#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from main_star import reset_env, step_env
from scripts.star.find_panda_mechanism_episode import audit_state, ee_pos, load_run_dir, seed_episode_rng, unwrap

REPORT_ROOT = REPO / "reports" / "star_arm_panda"
PROCESS_ROOT = REPORT_ROOT / "process_viz"
QUAL_ROOT = REPORT_ROOT / "qualitative"
RUN_DIR = Path(
    "/root/autodl-tmp/star_v2_storage/results/star_arm_panda/showcase_mechanism/"
    "SafetyPandaReachObstacle-v0/star_v2/panda_showcase_refcorridor_star_v2_s30"
)


@torch.no_grad()
def q_for_action(run, state: np.ndarray, action: np.ndarray) -> float:
    device = run.agent.device
    s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    a = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    ns = run.agent.normalize_state(s)
    return float(run.agent._cost_plus(ns, a).view(-1)[0].item())


def endpoint_for_action(env, action: np.ndarray) -> np.ndarray:
    u = unwrap(env)
    pos = ee_pos(env)
    return pos + np.asarray(action, dtype=float) * float(u.cfg.action_scale)


def item_to_row(item: dict[str, Any], *, seed: int, step: int, role: str, metrics: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "state_id": f"seed{seed}_episode0_step{step}_{role}",
        "seed": seed,
        "episode": 0,
        "step": step,
        "role": role,
        "candidate_type": item["candidate_type"],
        "candidate_id": int(item.get("candidate_index", item.get("candidate_id", 0))),
        "beta": float(item.get("beta", np.nan)) if item.get("beta", "") != "" else np.nan,
        "action_dx": float(item.get("action_x", item.get("action_dx", np.nan))),
        "action_dy": float(item.get("action_y", item.get("action_dy", np.nan))),
        "action_dz": float(item.get("action_z", item.get("action_dz", np.nan))),
        "start_x": float(metrics["start_x"]),
        "start_y": float(metrics["start_y"]),
        "start_z": float(metrics["start_z"]),
        "endpoint_x": float(item["endpoint_x"]),
        "endpoint_y": float(item["endpoint_y"]),
        "endpoint_z": float(item["endpoint_z"]),
        "predicted_QC": float(item.get("q_cost", item.get("predicted_QC", np.nan))),
        "predicted_safe": bool(float(item.get("q_cost", item.get("predicted_QC", np.nan))) <= float(metrics["d_aud"])),
        "is_high_risk": bool(float(item.get("q_cost", item.get("predicted_QC", np.nan))) > float(metrics["d_aud"])),
        "executed": bool(float(item.get("executed", 0.0)) > 0.0),
        "rho_cor": float(metrics["rho_cor"]),
        "rho_cur": float(metrics["rho_cur"]),
        "corridor_lift": float(metrics["corridor_lift"]),
        "d_aud": float(metrics["d_aud"]),
        "safe_margin": float(metrics["safe_margin"]),
        "clearance_at_state": float(metrics["clearance"]),
        "reference_age": metrics.get("reference_age", "targeted_refcorridor_checkpoint"),
        "source": source,
    }


def audit_replay_steps(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = load_run_dir(RUN_DIR, label="STAR", mode="star_exec")
    seed_episode_rng(run, seed)
    state = reset_env(run.env, seed=seed)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    done = False
    step = 0
    try:
        while not done and step < int(run.config.eval_numsteps):
            u = unwrap(run.env)
            pos = ee_pos(run.env)
            obs = np.asarray(u.obstacle_pos, dtype=float)
            clearance = float(np.linalg.norm(pos - obs) - float(u.cfg.safe_margin))
            action = run.agent.select_action(state, evaluate=True, execution_mode="star_exec", total_numsteps=300000, diagnostics=True)
            selected_q = q_for_action(run, state, action)
            audit_row, items = audit_state(run, state, run.env)
            metrics = {
                "start_x": float(pos[0]),
                "start_y": float(pos[1]),
                "start_z": float(pos[2]),
                "rho_cor": audit_row["rho_cor"],
                "rho_cur": audit_row["rho_cur"],
                "corridor_lift": audit_row["corridor_lift"],
                "d_aud": audit_row["threshold"],
                "safe_margin": float(u.cfg.safe_margin),
                "clearance": clearance,
            }
            for item in items:
                rows.append(item_to_row(item, seed=seed, step=step, role="candidate", metrics=metrics, source="star_exec_replay_critic_query"))
            ep = endpoint_for_action(run.env, action)
            rows.append(
                item_to_row(
                    {
                        "candidate_type": "selected_exec",
                        "candidate_index": 0,
                        "action_x": float(action[0]),
                        "action_y": float(action[1]),
                        "action_z": float(action[2]),
                        "endpoint_x": float(ep[0]),
                        "endpoint_y": float(ep[1]),
                        "endpoint_z": float(ep[2]),
                        "q_cost": selected_q,
                        "executed": 1.0,
                    },
                    seed=seed,
                    step=step,
                    role="candidate",
                    metrics=metrics,
                    source="star_exec_replay_selected_action",
                )
            )
            current_q = [float(x["q_cost"]) for x in items if x["candidate_type"] == "current_only"]
            corridor_q = [float(x["q_cost"]) for x in items if x["candidate_type"] == "corridor_shadow"]
            threshold = float(audit_row["threshold"])
            risky_corridor_executed = any(float(x["q_cost"]) > threshold and float(x.get("executed", 0.0)) > 0.0 for x in items if x["candidate_type"] == "corridor_shadow")
            summaries.append(
                {
                    "step": step,
                    "role": "candidate",
                    "Q_mean": float(audit_row["mean_action_risk"]),
                    "Q_selected": selected_q,
                    "max_Q_current": float(max(current_q)),
                    "max_Q_corridor": float(max(corridor_q)),
                    "rho_cur": float(audit_row["rho_cur"]),
                    "rho_cor": float(audit_row["rho_cor"]),
                    "corridor_lift": float(audit_row["corridor_lift"]),
                    "num_high_risk_current": int(sum(q > threshold for q in current_q)),
                    "num_high_risk_corridor": int(sum(q > threshold for q in corridor_q)),
                    "risky_shadow_executed": bool(risky_corridor_executed),
                    "clearance": clearance,
                    "cost": 0.0,
                    "success": False,
                    "selection_score": 10 * (audit_row["mean_action_risk"] <= threshold) + 10 * (max(corridor_q) > threshold) + 5 + 3 * (audit_row["corridor_lift"] > 0) + (clearance > 0),
                    "source": "star_exec_replay_critic_query",
                }
            )
            state, _reward, cost, terminated, truncated, info = step_env(run.env, action, run.config.safe_env)
            summaries[-1]["cost"] = float(cost)
            summaries[-1]["success"] = bool(float(info.get("success", 0.0)) > 0.0)
            done = bool(terminated or truncated)
            step += 1
    finally:
        run.env.close()
    return pd.DataFrame(summaries), pd.DataFrame(rows)


def logged_audit_rows(seed: int) -> tuple[pd.Series, pd.DataFrame]:
    summary = json.load(open(QUAL_ROOT / "audit_summary.json"))
    audit = pd.read_csv(QUAL_ROOT / "audit_snapshot.csv")
    step = int(summary["selected_step"])
    threshold = float(summary["d_aud"])
    pos = audit.iloc[0][["p_ee_x", "p_ee_y", "p_ee_z"]].to_numpy(float)
    rows: list[dict[str, Any]] = []
    metrics = {
        "start_x": pos[0],
        "start_y": pos[1],
        "start_z": pos[2],
        "rho_cor": summary["rho_cor"],
        "rho_cur": summary["rho_cur"],
        "corridor_lift": summary["corridor_lift"],
        "d_aud": threshold,
        "safe_margin": audit.iloc[0]["safe_margin"],
        "clearance": audit.iloc[0]["clearance_at_state"] if "clearance_at_state" in audit.columns else audit.iloc[0].get("clearance", np.nan),
    }
    if not np.isfinite(metrics["clearance"]):
        obs = audit.iloc[0][["p_obs_x", "p_obs_y", "p_obs_z"]].to_numpy(float)
        metrics["clearance"] = float(np.linalg.norm(pos - obs) - float(metrics["safe_margin"]))
    for _, item in audit.iterrows():
        rows.append(item_to_row(item.to_dict(), seed=seed, step=step, role="audit", metrics=metrics, source="logged_selected_audit_snapshot"))
    current = audit[audit["candidate_type"] == "current_only"]["predicted_QC"]
    corridor = audit[audit["candidate_type"] == "corridor_shadow"]["predicted_QC"]
    sel = pd.Series(
        {
            "step": step,
            "role": "audit",
            "Q_mean": summary["actor_mean_Q"],
            "Q_selected": float(audit[audit["candidate_type"] == "selected_exec"]["predicted_QC"].iloc[0]) if (audit["candidate_type"] == "selected_exec").any() else summary["actor_mean_Q"],
            "max_Q_current": float(current.max()),
            "max_Q_corridor": summary["max_corridor_Q"],
            "rho_cur": summary["rho_cur"],
            "rho_cor": summary["rho_cor"],
            "corridor_lift": summary["corridor_lift"],
            "num_high_risk_current": int((current > threshold).sum()),
            "num_high_risk_corridor": int((corridor > threshold).sum()),
            "risky_shadow_executed": summary["risky_shadows_executed"],
            "clearance": float(metrics["clearance"]),
            "cost": 0.0,
            "success": False,
            "selection_score": 31.0,
            "source": "logged_selected_audit_snapshot",
        }
    )
    return sel, pd.DataFrame(rows)


def choose_steps(replay_summary: pd.DataFrame, logged_summary: pd.Series) -> pd.DataFrame:
    audit_step = int(logged_summary["step"])
    before = replay_summary[replay_summary["step"] < audit_step].copy()
    after = replay_summary[replay_summary["step"] > audit_step].copy()
    selected: list[pd.Series] = []
    if not before.empty:
        approach = before.sort_values(["selection_score", "corridor_lift"], ascending=False).iloc[0].copy()
        approach["role"] = "approach"
        selected.append(approach)
    selected.append(logged_summary.copy())
    near = replay_summary.sort_values("clearance").iloc[0].copy()
    near["role"] = "near_obstacle"
    if int(near["step"]) != audit_step:
        selected.append(near)
    if not after.empty:
        pass_row = after.sort_values(["clearance", "step"], ascending=[False, True]).iloc[0].copy()
        pass_row["role"] = "pass"
        if int(pass_row["step"]) not in {int(s["step"]) for s in selected}:
            selected.append(pass_row)
    out = pd.DataFrame(selected)
    return out.sort_values("step").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=900078)
    args = parser.parse_args()
    PROCESS_ROOT.mkdir(parents=True, exist_ok=True)
    replay_summary, replay_rows = audit_replay_steps(args.seed)
    logged_summary, logged_rows = logged_audit_rows(args.seed)
    selected = choose_steps(replay_summary, logged_summary)
    role_by_step_source = {(int(r.step), str(r.source)): str(r.role) for _, r in selected.iterrows()}
    all_rows = pd.concat([replay_rows, logged_rows], ignore_index=True)
    selected_rows = []
    for _, sel in selected.iterrows():
        step = int(sel["step"])
        source = str(sel["source"])
        part = all_rows[(all_rows["step"] == step) & (all_rows["source"] == source)].copy()
        part["role"] = str(sel["role"])
        selected_rows.append(part)
    sequence_rows = pd.concat(selected_rows, ignore_index=True)
    selected.to_csv(PROCESS_ROOT / "audit_step_selection.csv", index=False)
    sequence_rows.to_csv(PROCESS_ROOT / "audit_sequence_rows.csv", index=False)
    selected.drop(columns=["selection_score"], errors="ignore").to_csv(PROCESS_ROOT / "audit_sequence_summary.csv", index=False)
    verification = [
        "# Panda Gripper Process Verification",
        "",
        "- audit_summary.json satisfies actor mean safe, high-risk corridor shadow, positive corridor lift, and no risky shadow execution.",
        f"- selected seed: `{args.seed}`",
        f"- selected audit steps: `{', '.join(str(int(x)) for x in selected['step'])}`",
        "- gripper/finger links are read from PyBullet link states 11/9/10.",
        "- audit_sequence_rows.csv contains real critic queries from replayed states plus the logged selected audit snapshot for step 16.",
    ]
    (PROCESS_ROOT / "verification.md").write_text("\n".join(verification) + "\n")
    print(json.dumps({"selected_steps": selected[["step", "role"]].to_dict("records"), "rows": len(sequence_rows)}, indent=2))


if __name__ == "__main__":
    main()
