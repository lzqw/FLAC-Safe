#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def fnum(value, default=math.nan) -> float:
    try:
        if value in ('', None):
            return default
        return float(value)
    except Exception:
        return default


def mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def sample_std(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return 0.0 if vals else math.nan
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def fmt(value: float) -> str:
    return '--' if math.isnan(value) else f'{value:.3f}'


def grouped_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if str(row.get('completed', '')).lower() not in {'true', '1'}:
            continue
        groups[(row.get('phase', ''), row.get('task', ''), row.get('method', ''))].append(row)
    out = []
    for (phase, task, method), vals in sorted(groups.items()):
        out.append({
            'phase': phase,
            'task': task,
            'method': method,
            'seeds': len(vals),
            'raw_return_mean': mean([fnum(v.get('raw_return_mean')) for v in vals]),
            'raw_return_std': sample_std([fnum(v.get('raw_return_mean')) for v in vals]),
            'raw_cost_mean': mean([fnum(v.get('raw_cost_mean')) for v in vals]),
            'raw_cost_std': sample_std([fnum(v.get('raw_cost_mean')) for v in vals]),
            'filtered_return_mean': mean([fnum(v.get('filtered_return_mean')) for v in vals]),
            'filtered_return_std': sample_std([fnum(v.get('filtered_return_mean')) for v in vals]),
            'filtered_cost_mean': mean([fnum(v.get('filtered_cost_mean')) for v in vals]),
            'filtered_cost_std': sample_std([fnum(v.get('filtered_cost_mean')) for v in vals]),
            'train_cost_rate_mean': mean([fnum(v.get('train_total_cost_rate')) for v in vals]),
        })
    return out


def latex_table(rows: list[dict], caption: str, label: str) -> str:
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        rf'\caption{{{caption}}}',
        rf'\label{{{label}}}',
        r'\begin{tabular}{lllrrrr}',
        r'\toprule',
        r'Phase & Task & Method & Seeds & Raw Ret. & Raw Cost & Train Cost Rate \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(
            f"{row['phase']} & {row['task']} & {row['method']} & {row['seeds']} & "
            f"{fmt(row['raw_return_mean'])} $\\pm$ {fmt(row['raw_return_std'])} & "
            f"{fmt(row['raw_cost_mean'])} $\\pm$ {fmt(row['raw_cost_std'])} & "
            f"{fmt(row['train_cost_rate_mean'])} \\\\"  # noqa: W605
        )
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    return '\n'.join(lines)


def build_claim_matrix(report_dir: Path, summary: list[dict]) -> str:
    core = [r for r in summary if r['phase'] == 'core_100k']
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in core:
        by_task[row['task']][row['method']] = row
    star_beats_current = []
    star_has_rows = []
    for task, methods in sorted(by_task.items()):
        star = methods.get('star_v2')
        cur = methods.get('current_only_v2')
        if star:
            star_has_rows.append(task)
        if star and cur and not math.isnan(star['raw_cost_mean']) and not math.isnan(cur['raw_cost_mean']):
            star_beats_current.append(star['raw_cost_mean'] < cur['raw_cost_mean'])
    def status(condition: bool, partial: bool = False) -> str:
        if condition:
            return 'SUPPORTED'
        return 'PARTIALLY SUPPORTED' if partial else 'NOT SUPPORTED'
    lines = ['# STAR-v2 Paper Claim Matrix', '']
    complete_tasks = len(by_task)
    lines.append('| Claim | Status | Evidence |')
    lines.append('| --- | --- | --- |')
    lines.append(f"| STAR-v2 core runs are available | {status(complete_tasks >= 4, complete_tasks > 0)} | tasks_with_core_rows={complete_tasks} |")
    lines.append(f"| STAR-v2 lowers raw cost versus current-only | {status(bool(star_beats_current) and all(star_beats_current), bool(star_beats_current))} | paired_task_cost_wins={sum(star_beats_current)}/{len(star_beats_current)} |")
    lines.append('| Candidate execution improves safety at same checkpoint | NOT SUPPORTED | requires executor/offline raw+filtered evaluation outputs |')
    lines.append('| Simulator oracle supports predicted shadow risk | NOT SUPPORTED | requires oracle_summary.csv |')
    lines.append('| Ablations identify corridor/log-mean-exp contribution | NOT SUPPORTED | requires ablation_100k completed rows |')
    lines.append('')
    lines.append('Statuses are generated only from currently selected completed/error-free CSV rows; missing data are not imputed.')
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-dir', default='reports/star_v2_final')
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    manifest = read_csv(report_dir / 'run_manifest.csv')
    summary = grouped_summary(manifest)
    tables = report_dir / 'tables'
    latex = report_dir / 'latex'
    figures = report_dir / 'figures'
    for path in (tables, latex, figures):
        path.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / 'summary_by_seed.csv', [r for r in manifest if str(r.get('completed', '')).lower() in {'true', '1'}])
    write_csv(report_dir / 'ablation_summary.csv', [r for r in summary if 'ablation' in r.get('phase', '')])
    write_csv(report_dir / 'efficiency_summary.csv', [{'note': 'wall-clock and overhead summaries require completed core/eval runs with timing columns'}])
    write_csv(report_dir / 'raw_vs_filtered.csv', summary)
    write_csv(report_dir / 'learning_curves.csv', [])
    write_csv(report_dir / 'mechanism_curves.csv', [])
    (report_dir / 'paper_claim_matrix.md').write_text(build_claim_matrix(report_dir, summary), encoding='utf-8')
    (latex / 'main_results.tex').write_text(latex_table(summary, 'STAR-v2 main results.', 'tab:star-v2-main'), encoding='utf-8')
    (latex / 'component_ablation.tex').write_text(latex_table([r for r in summary if 'ablation' in r['phase']], 'STAR-v2 component ablations.', 'tab:star-v2-components'), encoding='utf-8')
    (latex / 'shadow_design_ablation.tex').write_text(latex_table([r for r in summary if 'ablation' in r['phase']], 'STAR-v2 shadow design ablations.', 'tab:star-v2-shadow-design'), encoding='utf-8')
    (latex / 'efficiency.tex').write_text('% Efficiency table is generated after executor/oracle/core timing outputs are complete.\n', encoding='utf-8')
    for name in ['return_cost_pareto', 'raw_evr', 'predicted_svr_vs_executed_evr', 'hidden_unsafe_rate', 'boundary_safe_coverage', 'k_safety_overhead']:
        (figures / f'{name}.README.md').write_text(f'# {name}\n\nPending completed STAR-v2 final CSVs.\n', encoding='utf-8')
    (report_dir / 'latex' / 'experiments.tex').write_text(
        '% Auto-generated STAR-v2 experiment section stub.\n'
        '% Replace with full prose after core/resume/oracle/ablation gates are complete.\n'
        r'\input{reports/star_v2_final/latex/main_results.tex}' + '\n',
        encoding='utf-8',
    )
    trace_rows = []
    for row in summary:
        trace_rows.append({
            'artifact': 'latex/main_results.tex',
            'source_csv': 'run_manifest.csv',
            'phase': row['phase'],
            'task': row['task'],
            'method': row['method'],
        })
    write_csv(report_dir / 'result_traceability.csv', trace_rows)
    print(f'wrote STAR-v2 paper artifacts under {report_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
