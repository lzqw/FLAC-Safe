#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import struct
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable


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


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def sample_std(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 2:
        return 0.0 if vals else math.nan
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def fmt(value: float) -> str:
    return '--' if math.isnan(value) else f'{value:.3f}'


def completed_rows(manifest: list[dict]) -> list[dict]:
    return [r for r in manifest if str(r.get('completed', '')).lower() in {'true', '1'}]


def grouped_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in completed_rows(rows):
        groups[(row.get('phase', ''), row.get('task', ''), row.get('method', ''), row.get('ablation_name', ''))].append(row)
    out = []
    for (phase, task, method, ablation_name), vals in sorted(groups.items()):
        out.append({
            'phase': phase,
            'task': task,
            'method': method,
            'ablation_name': ablation_name,
            'seeds': len(vals),
            'raw_return_mean': mean(fnum(v.get('raw_return_mean')) for v in vals),
            'raw_return_std': sample_std(fnum(v.get('raw_return_mean')) for v in vals),
            'raw_cost_mean': mean(fnum(v.get('raw_cost_mean')) for v in vals),
            'raw_cost_std': sample_std(fnum(v.get('raw_cost_mean')) for v in vals),
            'raw_evr_mean': mean(fnum(v.get('raw_evr_mean')) for v in vals),
            'filtered_return_mean': mean(fnum(v.get('filtered_return_mean')) for v in vals),
            'filtered_return_std': sample_std(fnum(v.get('filtered_return_mean')) for v in vals),
            'filtered_cost_mean': mean(fnum(v.get('filtered_cost_mean')) for v in vals),
            'filtered_cost_std': sample_std(fnum(v.get('filtered_cost_mean')) for v in vals),
            'filtered_evr_mean': mean(fnum(v.get('filtered_evr_mean')) for v in vals),
            'train_cost_rate_mean': mean(fnum(v.get('train_total_cost_rate')) for v in vals),
        })
    return out


def index_rows(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    out = {}
    for row in completed_rows(rows):
        key = (row.get('phase', ''), row.get('task', ''), str(row.get('seed', '')), row.get('method', ''))
        out[key] = row
        ablation = row.get('ablation_name', '')
        if ablation:
            out[(row.get('phase', ''), row.get('task', ''), str(row.get('seed', '')), ablation)] = row
    return out


def bootstrap_ci(diffs: list[float], n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    vals = [d for d in diffs if not math.isnan(d)]
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        boots.append(mean(sample))
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return lo, hi


def paired_comparisons(rows: list[dict]) -> list[dict]:
    idx = index_rows(rows)
    specs = [
        ('STAR-Actor raw vs Pointwise raw', 'core_100k', 'star_v2', 'pointwise_v2', 'raw_cost_mean', 'lower'),
        ('STAR-Actor raw vs SAC-Lag-local raw', 'core_100k', 'star_v2', 'sac_lag', 'raw_cost_mean', 'lower'),
        ('Full STAR raw vs filtered', 'core_100k', 'star_v2:filtered', 'star_v2:raw', 'cost', 'lower'),
        ('Full STAR vs STAR-Exec', 'ablation_100k', 'full_star', 'candidate_execution_only', 'raw_cost_mean', 'lower'),
        ('corridor vs current-only', 'ablation_100k', 'corridor', 'current_only', 'raw_cost_mean', 'lower'),
        ('log-mean-exp vs mean', 'ablation_100k', 'agg_log_mean_exp', 'agg_mean', 'raw_cost_mean', 'lower'),
        ('log-mean-exp vs max', 'ablation_100k', 'agg_log_mean_exp', 'agg_max', 'raw_cost_mean', 'lower'),
    ]
    out = []
    for name, phase, left, right, metric, direction in specs:
        pairs = []
        tasks = sorted({r.get('task', '') for r in completed_rows(rows) if r.get('phase') == phase})
        seeds = sorted({str(r.get('seed', '')) for r in completed_rows(rows) if r.get('phase') == phase})
        diffs = []
        for task in tasks:
            for seed in seeds:
                if name == 'Full STAR raw vs filtered':
                    row = idx.get((phase, task, seed, 'star_v2'))
                    if not row:
                        continue
                    left_val = fnum(row.get('filtered_cost_mean'))
                    right_val = fnum(row.get('raw_cost_mean'))
                    left_id = right_id = row.get('run_name', '')
                else:
                    lrow = idx.get((phase, task, seed, left))
                    rrow = idx.get((phase, task, seed, right))
                    if not lrow or not rrow:
                        continue
                    left_val = fnum(lrow.get(metric))
                    right_val = fnum(rrow.get(metric))
                    left_id = lrow.get('run_name', '')
                    right_id = rrow.get('run_name', '')
                if math.isnan(left_val) or math.isnan(right_val):
                    continue
                diff = left_val - right_val
                diffs.append(diff)
                pairs.append(f'{task}:seed{seed}:{left_id}->{right_id}')
        lo, hi = bootstrap_ci(diffs)
        out.append({
            'comparison': name,
            'phase': phase,
            'metric': metric,
            'direction': direction,
            'n_pairs': len(diffs),
            'mean_diff_left_minus_right': mean(diffs),
            'sample_std_diff': sample_std(diffs),
            'bootstrap95_ci_low': lo,
            'bootstrap95_ci_high': hi,
            'supported_direction': bool(diffs) and (hi < 0 if direction == 'lower' else lo > 0),
            'source_pairs': ';'.join(pairs),
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


def build_claim_matrix(summary: list[dict], comparisons: list[dict]) -> str:
    def comp(name: str) -> dict | None:
        return next((r for r in comparisons if r['comparison'] == name), None)
    def status(condition: bool, partial: bool = False) -> str:
        if condition:
            return 'SUPPORTED'
        return 'PARTIALLY SUPPORTED' if partial else 'NOT SUPPORTED'
    core_tasks = {r['task'] for r in summary if r['phase'] == 'core_100k'}
    star_pointwise = comp('STAR-Actor raw vs Pointwise raw')
    star_lag = comp('STAR-Actor raw vs SAC-Lag-local raw')
    raw_filtered = comp('Full STAR raw vs filtered')
    corridor_current = comp('corridor vs current-only')
    logmean_mean = comp('log-mean-exp vs mean')
    logmean_max = comp('log-mean-exp vs max')
    lines = ['# STAR-v2 Paper Claim Matrix', '']
    lines.append('| Claim | Status | Evidence |')
    lines.append('| --- | --- | --- |')
    lines.append(f"| Core STAR-v2 rows are available | {status(len(core_tasks) >= 4, len(core_tasks) > 0)} | tasks_with_core_rows={len(core_tasks)} |")
    lines.append(f"| STAR-v2 raw actor improves safety over Pointwise/SAC-Lag-local | {status(bool(star_pointwise and star_pointwise['supported_direction']) and bool(star_lag and star_lag['supported_direction']), bool((star_pointwise and star_pointwise['n_pairs']) or (star_lag and star_lag['n_pairs'])))} | pointwise_pairs={star_pointwise['n_pairs'] if star_pointwise else 0}; saclag_pairs={star_lag['n_pairs'] if star_lag else 0} |")
    lines.append(f"| Candidate execution improves same-checkpoint safety | {status(bool(raw_filtered and raw_filtered['supported_direction']), bool(raw_filtered and raw_filtered['n_pairs']))} | raw_vs_filtered_pairs={raw_filtered['n_pairs'] if raw_filtered else 0} |")
    lines.append('| Simulator oracle supports predicted shadow risk | NOT SUPPORTED | requires oracle_summary.csv and precision/recall/AUROC thresholds |')
    design_supported = bool(corridor_current and corridor_current['supported_direction']) and bool(logmean_mean and logmean_mean['supported_direction'] or logmean_max and logmean_max['supported_direction'])
    design_partial = bool((corridor_current and corridor_current['n_pairs']) or (logmean_mean and logmean_mean['n_pairs']) or (logmean_max and logmean_max['n_pairs']))
    lines.append(f"| Corridor/log-mean-exp design choices are empirically supported | {status(design_supported, design_partial)} | corridor_pairs={corridor_current['n_pairs'] if corridor_current else 0}; logmean_pairs={(logmean_mean['n_pairs'] if logmean_mean else 0) + (logmean_max['n_pairs'] if logmean_max else 0)} |")
    lines.append('')
    lines.append('Statuses use only selected completed/error-free CSV rows. Missing comparisons are not imputed.')
    return '\n'.join(lines) + '\n'




def write_placeholder_png(path: Path, title: str, width: int = 900, height: int = 560) -> None:
    # Minimal valid RGB PNG using only the standard library. This is a fallback
    # for cluster environments without matplotlib; it intentionally encodes a
    # plain white canvas so the expected artifact path exists and is traceable.
    raw_rows = []
    for _ in range(height):
        raw_rows.append(b'\x00' + b'\xff\xff\xff' * width)
    raw = b''.join(raw_rows)
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    png += chunk(b'iTXt', b'Title\x00\x00\x00\x00\x00' + title.encode('utf-8', errors='ignore'))
    png += chunk(b'IDAT', zlib.compress(raw, level=9))
    png += chunk(b'IEND', b'')
    path.write_bytes(png)


def write_placeholder_pdf(path: Path, title: str) -> None:
    text = f'{title} - pending completed STAR-v2 data'.replace('(', '[').replace(')', ']')
    stream = f'BT /F1 18 Tf 72 720 Td ({text}) Tj ET'
    objects = [
        '1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj',
        '2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj',
        '3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj',
        '4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj',
        f'5 0 obj << /Length {len(stream)} >> stream\n{stream}\nendstream endobj',
    ]
    content = '%PDF-1.4\n'
    offsets = [0]
    for obj in objects:
        offsets.append(len(content.encode('latin-1')))
        content += obj + '\n'
    xref_pos = len(content.encode('latin-1'))
    content += f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'
    for off in offsets[1:]:
        content += f'{off:010d} 00000 n \n'
    content += f'trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n'
    path.write_bytes(content.encode('latin-1'))

def write_figures(fig_dir: Path, summary: list[dict], comparisons: list[dict]) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        for name in ['return_cost_pareto', 'raw_evr', 'predicted_svr_vs_executed_evr', 'hidden_unsafe_rate', 'boundary_safe_coverage', 'k_safety_overhead']:
            title = name.replace('_', ' ').title()
            write_placeholder_png(fig_dir / f'{name}.png', title)
            write_placeholder_pdf(fig_dir / f'{name}.pdf', title)
            (fig_dir / f'{name}.README.md').write_text(f'Placeholder generated because matplotlib is unavailable: {exc}\n', encoding='utf-8')
        return

    def save(name: str, build):
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        build(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / f'{name}.png', dpi=200)
        fig.savefig(fig_dir / f'{name}.pdf')
        plt.close(fig)

    def no_data(ax, title: str):
        ax.text(0.5, 0.5, 'No completed data yet', ha='center', va='center')
        ax.set_title(title)
        ax.set_axis_off()

    save('return_cost_pareto', lambda ax: (
        no_data(ax, 'Return-cost Pareto') if not summary else [
            ax.scatter([r['raw_cost_mean'] for r in summary], [r['raw_return_mean'] for r in summary]),
            ax.set_xlabel('Episode cost'), ax.set_ylabel('Return'), ax.set_title('Return-cost Pareto')
        ]
    ))
    save('raw_evr', lambda ax: (
        no_data(ax, 'Raw EVR') if not summary else [
            ax.bar(range(len(summary)), [0 if math.isnan(r['raw_evr_mean']) else r['raw_evr_mean'] for r in summary]),
            ax.set_ylabel('Raw EVR'), ax.set_title('Raw executed violation rate')
        ]
    ))
    for name, title in [
        ('predicted_svr_vs_executed_evr', 'Predicted SVR vs executed EVR'),
        ('hidden_unsafe_rate', 'Hidden unsafe rate'),
        ('boundary_safe_coverage', 'Boundary safe coverage'),
        ('k_safety_overhead', 'K safety/overhead'),
    ]:
        save(name, lambda ax, t=title: no_data(ax, t))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-dir', default='reports/star_v2_final')
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    manifest = read_csv(report_dir / 'run_manifest.csv')
    summary = grouped_summary(manifest)
    comparisons = paired_comparisons(manifest)
    tables = report_dir / 'tables'
    latex = report_dir / 'latex'
    figures = report_dir / 'figures'
    for path in (tables, latex, figures):
        path.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / 'summary_by_seed.csv', completed_rows(manifest))
    write_csv(report_dir / 'paired_comparisons.csv', comparisons)
    write_csv(report_dir / 'ablation_summary.csv', [r for r in summary if 'ablation' in r.get('phase', '')])
    write_csv(report_dir / 'efficiency_summary.csv', [{'note': 'wall-clock and overhead summaries require completed core/eval runs with timing columns'}])
    write_csv(report_dir / 'raw_vs_filtered.csv', summary)
    write_csv(report_dir / 'learning_curves.csv', [])
    write_csv(report_dir / 'mechanism_curves.csv', [])
    (report_dir / 'paper_claim_matrix.md').write_text(build_claim_matrix(summary, comparisons), encoding='utf-8')
    (latex / 'main_results.tex').write_text(latex_table([r for r in summary if r['phase'] in {'core_100k', 'resume_300k'}], 'STAR-v2 main results.', 'tab:star-v2-main'), encoding='utf-8')
    (latex / 'component_ablation.tex').write_text(latex_table([r for r in summary if 'ablation' in r['phase']], 'STAR-v2 component ablations.', 'tab:star-v2-components'), encoding='utf-8')
    (latex / 'shadow_design_ablation.tex').write_text(latex_table([r for r in summary if 'ablation' in r['phase']], 'STAR-v2 shadow design ablations.', 'tab:star-v2-shadow-design'), encoding='utf-8')
    (latex / 'efficiency.tex').write_text('% Efficiency table is generated after executor/oracle/core timing outputs are complete.\n', encoding='utf-8')
    write_figures(figures, summary, comparisons)
    (latex / 'experiments.tex').write_text(
        '% Auto-generated STAR-v2 experiment section stub.\n'
        '% Predicted shadow violations are critic-scored counterfactuals; executed violations are environment costs from selected actions.\n'
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
            'ablation_name': row.get('ablation_name', ''),
        })
    for row in comparisons:
        trace_rows.append({
            'artifact': 'paired_comparisons.csv',
            'source_csv': 'run_manifest.csv',
            'phase': row['phase'],
            'task': 'paired',
            'method': row['comparison'],
            'ablation_name': row.get('source_pairs', ''),
        })
    write_csv(report_dir / 'result_traceability.csv', trace_rows)
    print(f'wrote STAR-v2 paper artifacts under {report_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
