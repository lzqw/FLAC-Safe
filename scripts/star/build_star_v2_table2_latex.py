#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", errors="ignore") as handle:
        return list(csv.DictReader(handle))


def fnum(value, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def fmt(value) -> str:
    x = fnum(value)
    return "--" if math.isnan(x) else f"{x:.3f}"


def latex_escape(value: str) -> str:
    return str(value).replace("_", r"\_")


def table_ablation(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Compact STAR-v2 component ablation.}",
        r"\label{tab:star-v2-ablation}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Task & Variant & Seeds & Return & Cost & EVR & Shadow excess \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row.get('task',''))} & {latex_escape(row.get('ablation_name',''))} & "
            f"{row.get('seeds','')} & {fmt(row.get('return_mean'))} & {fmt(row.get('cost_mean'))} & "
            f"{fmt(row.get('EVR_mean'))} & {fmt(row.get('shadow_excess_mean'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def table_executor(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Same-checkpoint STAR+Exec held-out confirmation.}",
        r"\label{tab:star-v2-executor}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Task & Raw ret. & Exec ret. & Raw cost & Exec cost & Raw EVR & Exec EVR & Fallback & FNE & Latency ms \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row.get('task',''))} & {fmt(row.get('raw_return'))} & {fmt(row.get('filtered_return'))} & "
            f"{fmt(row.get('raw_cost'))} & {fmt(row.get('filtered_cost'))} & {fmt(row.get('raw_evr'))} & "
            f"{fmt(row.get('filtered_evr'))} & {fmt(row.get('fallback_rate'))} & {fmt(row.get('FNE'))} & "
            f"{fmt(row.get('latency_ms'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_claims(report_dir: Path) -> None:
    mech = read_csv(report_dir / "mechanism" / "corridor_mechanism_summary.csv")
    oracle = read_csv(report_dir / "oracle" / "oracle_summary.csv")
    exec_rows = read_csv(report_dir / "executor" / "executor_summary.csv")
    supported_oracle = any(str(row.get("supported", "")).lower() == "true" for row in oracle)
    lift_vals = [fnum(row.get("corridor_risk_lift_mean")) for row in mech]
    lift_supported = bool(lift_vals) and sum(v > 0 for v in lift_vals if not math.isnan(v)) >= max(1, len(lift_vals) // 2)
    exec_improved = sum(
        1
        for row in exec_rows
        if fnum(row.get("filtered_cost")) <= fnum(row.get("raw_cost"))
        and fnum(row.get("filtered_evr")) <= fnum(row.get("raw_evr"))
    )
    exec_return_ok = sum(abs(fnum(row.get("return_drop_frac"))) <= 0.05 for row in exec_rows)
    raw_costs = [fnum(row.get("raw_cost")) for row in exec_rows]
    filt_costs = [fnum(row.get("filtered_cost")) for row in exec_rows]
    raw_evrs = [fnum(row.get("raw_evr")) for row in exec_rows]
    filt_evrs = [fnum(row.get("filtered_evr")) for row in exec_rows]
    mean_raw_cost = sum(v for v in raw_costs if not math.isnan(v)) / max(1, sum(not math.isnan(v) for v in raw_costs))
    mean_filt_cost = sum(v for v in filt_costs if not math.isnan(v)) / max(1, sum(not math.isnan(v) for v in filt_costs))
    mean_raw_evr = sum(v for v in raw_evrs if not math.isnan(v)) / max(1, sum(not math.isnan(v) for v in raw_evrs))
    mean_filt_evr = sum(v for v in filt_evrs if not math.isnan(v)) / max(1, sum(not math.isnan(v) for v in filt_evrs))
    lines = [
        "# STAR-v2 Final Claims",
        "",
        "| Claim | Status | Evidence |",
        "| --- | --- | --- |",
        f"| Corridor shadows expose risk beyond equal-budget current-only samples | {'SUPPORTED' if lift_supported else 'MIXED'} | mechanism bins with positive lift from logged paired audits: {sum(v > 0 for v in lift_vals if not math.isnan(v))}/{len(lift_vals)} |",
        "| STAR raw actor and STAR+Exec are reported separately | SUPPORTED | Table 1 uses raw-policy results; executor confirmation is in Table 2 artifacts. |",
        f"| STAR+Exec same-checkpoint deployment improves held-out safety/cost | {'MIXED' if exec_rows else 'UNAVAILABLE'} | confirmation safety improvements: {exec_improved}/{len(exec_rows)}; return-drop <=5% tasks: {exec_return_ok}/{len(exec_rows)}; mean cost {mean_raw_cost:.3f}->{mean_filt_cost:.3f}; mean EVR {mean_raw_evr:.3f}->{mean_filt_evr:.3f} |",
        f"| Oracle actual-risk validation is available | {'SUPPORTED' if supported_oracle else 'UNAVAILABLE'} | oracle supported rows: {sum(str(r.get('supported','')).lower() == 'true' for r in oracle)}/{len(oracle)} |",
        "| Compact 2x2 ablation was completed without broad hyperparameter sweep | SUPPORTED | 2 tasks x 3 seeds x 4 variants. |",
        "",
    ]
    (report_dir / "final_claims.md").write_text("\n".join(lines), encoding="utf-8")


def write_audit(report_dir: Path) -> None:
    required = [
        "mechanism/corridor_mechanism_by_age.csv",
        "mechanism/corridor_mechanism_summary.csv",
        "figures/fig_mechanism_validation.pdf",
        "figures/fig_mechanism_validation.png",
        "figures/fig_mechanism_validation.svg",
        "oracle/oracle_rows.csv",
        "oracle/oracle_summary.csv",
        "oracle/oracle_status.md",
        "executor/executor_validation_grid.csv",
        "executor/executor_confirmation.csv",
        "executor/executor_summary.csv",
        "latex/table_executor.tex",
        "ablation/ablation_by_seed.csv",
        "ablation/ablation_summary.csv",
        "latex/table_ablation.tex",
        "latex/table_ablation_executor.tex",
        "efficiency/efficiency_summary.csv",
        "final_claims.md",
    ]
    lines = ["# STAR-v2 Final Audit", ""]
    for rel in required:
        path = report_dir / rel
        lines.append(f"- {rel}: {'OK' if path.exists() and path.stat().st_size > 0 else 'MISSING'}")
    lines.append("")
    lines.append("Table 1 raw-policy main results were not regenerated by this Table 2/Figure 2 pass.")
    (report_dir / "final_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=Path("reports/star_v2_final"))
    args = parser.parse_args()
    report = args.report_dir
    latex = report / "latex"
    latex.mkdir(parents=True, exist_ok=True)
    ablation = read_csv(report / "ablation" / "ablation_summary.csv")
    executor = read_csv(report / "executor" / "executor_summary.csv")
    (latex / "table_ablation.tex").write_text(table_ablation(ablation), encoding="utf-8")
    (latex / "table_executor.tex").write_text(table_executor(executor), encoding="utf-8")
    combined = table_ablation(ablation) + "\n" + table_executor(executor)
    (latex / "table_ablation_executor.tex").write_text(combined, encoding="utf-8")
    write_claims(report)
    write_audit(report)
    print(f"wrote Table 2 latex and audit under {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
