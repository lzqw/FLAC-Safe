from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


class ShadowOracleUnsupported(RuntimeError):
    """Raised when an environment cannot be snapshotted reliably."""


@dataclass
class MujocoSnapshot:
    task: Any
    model: Any
    data: Any
    qpos: np.ndarray
    qvel: np.ndarray
    time: float
    ctrl: Optional[np.ndarray]
    act: Optional[np.ndarray]
    qacc_warmstart: Optional[np.ndarray]
    mocap_pos: Optional[np.ndarray]
    mocap_quat: Optional[np.ndarray]
    wrapper_attrs: list[tuple[Any, str, Any]]


def _wrapper_chain(env) -> list[Any]:
    chain = []
    current = env
    for _ in range(64):
        chain.append(current)
        if not hasattr(current, "env"):
            break
        current = current.env
    return chain


def _safe_scalar(value):
    if isinstance(value, (int, float, bool, str, type(None), np.integer, np.floating, np.bool_)):
        return value.item() if hasattr(value, "item") else value
    return None


def _get_task(env):
    base = getattr(env, "unwrapped", env)
    task = getattr(base, "task", None)
    if task is None:
        raise ShadowOracleUnsupported("env.unwrapped.task is unavailable")
    return task


def snapshot_supported(env) -> tuple[bool, str]:
    try:
        task = _get_task(env)
        model = getattr(task, "model")
        data = getattr(task, "data")
        if model is None or data is None:
            return False, "MuJoCo model/data are unavailable; call env.reset() before diagnostics"
        for name in ("qpos", "qvel"):
            if not hasattr(data, name):
                return False, f"MuJoCo data.{name} is unavailable"
        try:
            import mujoco  # noqa: F401
        except Exception as exc:  # pragma: no cover
            return False, f"mujoco Python module is unavailable: {exc}"
        return True, "supported"
    except Exception as exc:
        return False, str(exc)


def capture_snapshot(env) -> MujocoSnapshot:
    supported, reason = snapshot_supported(env)
    if not supported:
        raise ShadowOracleUnsupported(reason)

    task = _get_task(env)
    model = task.model
    data = task.data
    wrapper_attrs = []
    for wrapper in _wrapper_chain(env):
        for attr in ("_elapsed_steps", "elapsed_steps", "_has_reset"):
            if hasattr(wrapper, attr):
                value = _safe_scalar(getattr(wrapper, attr))
                if value is not None:
                    wrapper_attrs.append((wrapper, attr, value))

    def maybe_copy(name):
        value = getattr(data, name, None)
        return None if value is None else np.array(value, copy=True)

    return MujocoSnapshot(
        task=task,
        model=model,
        data=data,
        qpos=np.array(data.qpos, copy=True),
        qvel=np.array(data.qvel, copy=True),
        time=float(data.time),
        ctrl=maybe_copy("ctrl"),
        act=maybe_copy("act"),
        qacc_warmstart=maybe_copy("qacc_warmstart"),
        mocap_pos=maybe_copy("mocap_pos"),
        mocap_quat=maybe_copy("mocap_quat"),
        wrapper_attrs=wrapper_attrs,
    )


def restore_snapshot(snapshot: MujocoSnapshot) -> None:
    data = snapshot.data
    data.qpos[:] = snapshot.qpos
    data.qvel[:] = snapshot.qvel
    data.time = snapshot.time
    if snapshot.ctrl is not None and hasattr(data, "ctrl"):
        data.ctrl[:] = snapshot.ctrl
    if snapshot.act is not None and hasattr(data, "act") and len(snapshot.act) == len(data.act):
        data.act[:] = snapshot.act
    if snapshot.qacc_warmstart is not None and hasattr(data, "qacc_warmstart"):
        data.qacc_warmstart[:] = snapshot.qacc_warmstart
    if snapshot.mocap_pos is not None and hasattr(data, "mocap_pos"):
        data.mocap_pos[:] = snapshot.mocap_pos
    if snapshot.mocap_quat is not None and hasattr(data, "mocap_quat"):
        data.mocap_quat[:] = snapshot.mocap_quat
    for wrapper, attr, value in snapshot.wrapper_attrs:
        try:
            setattr(wrapper, attr, value)
        except Exception:
            pass

    import mujoco

    mujoco.mj_forward(snapshot.model, data)


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 6:
        next_state, reward, cost, terminated, truncated, info = out
        return next_state, float(reward), float(cost), bool(terminated), bool(truncated), info
    next_state, reward, terminated, truncated, info = out
    return next_state, float(reward), float(info.get("cost", 0.0)), bool(terminated), bool(truncated), info


def _state_tensor(agent, state):
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    return agent.normalize_state(state_tensor)


@torch.no_grad()
def _mean_action(agent, state) -> np.ndarray:
    state_tensor = _state_tensor(agent, state)
    _, _, mean_action = agent.policy.sample(state_tensor)
    action = mean_action.cpu().numpy()[0]
    return np.clip(action, agent.action_space.low, agent.action_space.high)


@torch.no_grad()
def _risk(agent, state, action) -> float:
    state_tensor = _state_tensor(agent, state)
    action_tensor = torch.as_tensor(action, dtype=torch.float32, device=agent.device).view(1, -1)
    q1, q2 = agent.cost_critic(state_tensor, action_tensor)
    value = agent.reduce_cost_values(q1, q2)
    return float(value.view(-1)[0].detach().cpu().item())


@torch.no_grad()
def _highest_risk_shadow_action(agent, state, k: Optional[int] = None) -> tuple[np.ndarray, float]:
    state_tensor = _state_tensor(agent, state)
    count = int(k or max(getattr(agent.audit, "shadow_k", 1), getattr(agent, "star_exec_candidates", 1)))
    shadow = agent.audit.generate_shadow_actions(agent.policy, agent.reference_policy, state_tensor, k=count)
    q_shadow = agent.audit.conservative_cost(agent.cost_critic, state_tensor, shadow.actions).view(-1)
    index = int(torch.argmax(q_shadow).item())
    action = shadow.actions.view(count, -1)[index].detach().cpu().numpy()
    action = np.clip(action, agent.action_space.low, agent.action_space.high)
    return action, float(q_shadow[index].detach().cpu().item())


def _executed_candidate_action(agent, state, total_numsteps: Optional[int]) -> tuple[np.ndarray, float]:
    previous_info = dict(getattr(agent, "last_action_info", {}) or {})
    try:
        action = agent.select_action(
            state,
            evaluate=True,
            execution_mode="star_exec",
            total_numsteps=total_numsteps,
        )
        risk = float(getattr(agent, "last_action_info", {}).get("selected_predicted_risk", _risk(agent, state, action)))
        return np.asarray(action, dtype=np.float32), risk
    finally:
        agent.last_action_info = previous_info


def _rollout_actual_cost(agent, env, snapshot: MujocoSnapshot, first_action, horizon: int) -> tuple[float, float]:
    restore_snapshot(snapshot)
    state, _, cost, terminated, truncated, _ = _step_env(env, first_action)
    total_cost = float(cost)
    violation = float(cost > 0)
    done = terminated or truncated
    for _ in range(max(0, int(horizon) - 1)):
        if done:
            break
        action = _mean_action(agent, state)
        state, _, cost, terminated, truncated, _ = _step_env(env, action)
        total_cost += float(cost)
        violation = max(violation, float(cost > 0))
        done = terminated or truncated
    restore_snapshot(snapshot)
    return violation, total_cost


def shadow_oracle_step(
    *,
    agent,
    env,
    state,
    total_numsteps: Optional[int] = None,
    horizon: int = 1,
    threshold: Optional[float] = None,
    shadow_k: Optional[int] = None,
) -> Dict[str, float | int | bool | str]:
    """Evaluation-only simulator oracle for STAR shadow actions.

    This function snapshots the evaluation environment, probes actor mean,
    executed candidate, and highest predicted-risk shadow action, then restores
    the simulator state. It does not add replay samples, increment training
    counters, update networks, or belong in the main training loop.
    """

    threshold = float(agent.risk_threshold if threshold is None else threshold)
    supported, reason = snapshot_supported(env)
    if not supported:
        return {
            "evaluation_only": True,
            "supported": False,
            "unsupported_reason": reason,
        }

    snapshot = capture_snapshot(env)
    previous_info = dict(getattr(agent, "last_action_info", {}) or {})
    try:
        mean_action = _mean_action(agent, state)
        executed_action, executed_predicted_risk = _executed_candidate_action(agent, state, total_numsteps)
        shadow_action, shadow_predicted_risk = _highest_risk_shadow_action(agent, state, k=shadow_k)
        mean_predicted_risk = _risk(agent, state, mean_action)

        mean_violation, mean_cost = _rollout_actual_cost(agent, env, snapshot, mean_action, horizon)
        executed_violation, executed_cost = _rollout_actual_cost(agent, env, snapshot, executed_action, horizon)
        shadow_violation, shadow_cost = _rollout_actual_cost(agent, env, snapshot, shadow_action, horizon)
    finally:
        restore_snapshot(snapshot)
        agent.last_action_info = previous_info

    unsafe_found_but_not_deployed = (
        shadow_predicted_risk > threshold
        and not np.allclose(shadow_action, executed_action)
        and executed_violation <= 0
    )

    return {
        "evaluation_only": True,
        "supported": True,
        "unsupported_reason": "",
        "horizon": int(horizon),
        "mean_predicted_risk": mean_predicted_risk,
        "executed_predicted_risk": executed_predicted_risk,
        "shadow_predicted_risk": shadow_predicted_risk,
        "mean_actual_cost": mean_cost,
        "executed_actual_cost": executed_cost,
        "shadow_actual_cost": shadow_cost,
        "mean_violation": mean_violation,
        "executed_violation": executed_violation,
        "shadow_violation": shadow_violation,
        "shadow_predicted_unsafe": float(shadow_predicted_risk > threshold),
        "executed_predicted_unsafe": float(executed_predicted_risk > threshold),
        "unsafe_found_but_not_deployed": float(unsafe_found_but_not_deployed),
    }


def _auroc(scores: Iterable[float], labels: Iterable[float]) -> float:
    positives = [s for s, y in zip(scores, labels) if y > 0]
    negatives = [s for s, y in zip(scores, labels) if y <= 0]
    if not positives or not negatives:
        return float("nan")
    total = 0.0
    count = 0
    for p in positives:
        for n in negatives:
            total += 1.0 if p > n else 0.5 if p == n else 0.0
            count += 1
    return total / max(1, count)


def _pearson(x: Iterable[float], y: Iterable[float]) -> float:
    xs = np.asarray(list(x), dtype=np.float64)
    ys = np.asarray(list(y), dtype=np.float64)
    if xs.size < 2 or ys.size < 2 or np.std(xs) <= 1e-12 or np.std(ys) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


class ShadowOracleAccumulator:
    """Aggregate evaluation-only shadow oracle rows."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self.rows: list[Dict[str, Any]] = []

    def add(self, row: Dict[str, Any]) -> None:
        if row.get("supported", False):
            self.rows.append(row)

    def summary(self) -> Dict[str, float | int | bool]:
        if not self.rows:
            return {
                "evaluation_only": True,
                "oracle_samples": 0,
                "oracle_shadow_violation_rate": float("nan"),
                "oracle_executed_violation_rate": float("nan"),
                "shadow_risk_precision": float("nan"),
                "shadow_risk_recall": float("nan"),
                "shadow_risk_AUROC": float("nan"),
                "predicted_vs_actual_shadow_cost": float("nan"),
                "unsafe_found_but_not_deployed_rate": float("nan"),
            }
        scores = [float(row["shadow_predicted_risk"]) for row in self.rows]
        labels = [float(row["shadow_violation"]) for row in self.rows]
        predicted = [float(row["shadow_predicted_unsafe"]) for row in self.rows]
        tp = sum(1.0 for p, y in zip(predicted, labels) if p > 0 and y > 0)
        fp = sum(1.0 for p, y in zip(predicted, labels) if p > 0 and y <= 0)
        fn = sum(1.0 for p, y in zip(predicted, labels) if p <= 0 and y > 0)
        return {
            "evaluation_only": True,
            "oracle_samples": len(self.rows),
            "oracle_shadow_violation_rate": float(np.mean(labels)),
            "oracle_executed_violation_rate": float(np.mean([float(row["executed_violation"]) for row in self.rows])),
            "shadow_risk_precision": tp / max(1.0, tp + fp),
            "shadow_risk_recall": tp / max(1.0, tp + fn),
            "shadow_risk_AUROC": _auroc(scores, labels),
            "predicted_vs_actual_shadow_cost": _pearson(scores, [float(row["shadow_actual_cost"]) for row in self.rows]),
            "unsafe_found_but_not_deployed_rate": float(
                np.mean([float(row["unsafe_found_but_not_deployed"]) for row in self.rows])
            ),
        }
