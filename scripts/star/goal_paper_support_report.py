#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from statistics import mean, stdev

MAIN = Path('reports/star_paper_support_300k/summary_by_seed.csv')
EXEC_GRID = Path('reports/star_paper_support/executor_grid/executor_grid_summary.csv')
CORRIDOR = Path('reports/star_goal/corridor_vs_current_only.csv')
ORACLE = Path('reports/star_paper_support/oracle_summary.csv')
OUT = Path('reports/star_paper_support')
STAR_GOAL = Path('reports/star_goal')
CONFIGS = Path('configs')


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def f(row: dict, key: str, default: float = math.nan) -> float:
    try:
        v = row.get(key, '')
        return float(v) if v != '' else default
    except Exception:
        return default


def selected_rows() -> list[dict[str, str]]:
    rows = read_csv(MAIN)
    return [r for r in rows if r.get('completed') == 'True' and r.get('has_error') == 'False']


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], dict[str, str]]:
    out = {}
    for r in rows:
        try:
            out[(r['task'], r['method'], int(r['seed']))] = r
        except Exception:
            pass
    return out


def copy_configs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copies = [
        (STAR_GOAL / 'selected_actor_config.json', OUT / 'selected_actor_config.json'),
        (STAR_GOAL / 'selected_baseline_config.json', OUT / 'selected_baselines.json'),
        (CONFIGS / 'star_selected_executor.json', OUT / 'selected_executor_config.json'),
        (STAR_GOAL / 'corridor_vs_current_only.csv', OUT / 'corridor_vs_current_only.csv'),
        (STAR_GOAL / 'corridor_vs_current_only.md', OUT / 'corridor_vs_current_only.md'),
        (Path('reports/star_paper_support/final/run_manifest.csv'), OUT / 'run_manifest.csv'),
        (EXEC_GRID, OUT / 'executor_grid.csv'),
        (ORACLE, OUT / 'oracle_summary.csv'),
    ]
    for src, dst in copies:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dst.resolve():
                shutil.copyfile(src, dst)


def raw_actor_comparison(rows: list[dict[str, str]]) -> list[dict]:
    data = by_key(rows)
    out = []
    for task in sorted({r['task'] for r in rows}):
        for seed in sorted({int(r['seed']) for r in rows if r['task'] == task}):
            star = data.get((task, 'star_actor', seed))
            if not star:
                continue
            for base in ['pointwise', 'sac_lag']:
                b = data.get((task, base, seed))
                if not b:
                    continue
                out.append({
                    'task': task,
                    'seed': seed,
                    'comparison': f'star_actor_vs_{base}',
                    'star_actor_raw_return': f(star, 'raw_return'),
                    'baseline_raw_return': f(b, 'raw_return'),
                    'delta_return_star_minus_baseline': f(star, 'raw_return') - f(b, 'raw_return'),
                    'star_actor_raw_cost': f(star, 'raw_episode_cost'),
                    'baseline_raw_cost': f(b, 'raw_episode_cost'),
                    'delta_cost_star_minus_baseline': f(star, 'raw_episode_cost') - f(b, 'raw_episode_cost'),
                    'star_actor_raw_evr': f(star, 'raw_EVR'),
                    'baseline_raw_evr': f(b, 'raw_EVR'),
                    'delta_evr_star_minus_baseline': f(star, 'raw_EVR') - f(b, 'raw_EVR'),
                    'star_actor_pSVR': f(star, 'pSVR'),
                    'star_actor_hidden_unsafe_rate': f(star, 'hidden_unsafe_rate'),
                })
    fields = ['task','seed','comparison','star_actor_raw_return','baseline_raw_return','delta_return_star_minus_baseline','star_actor_raw_cost','baseline_raw_cost','delta_cost_star_minus_baseline','star_actor_raw_evr','baseline_raw_evr','delta_evr_star_minus_baseline','star_actor_pSVR','star_actor_hidden_unsafe_rate']
    write_csv(OUT / 'raw_actor_comparison.csv', out, fields)
    return out


def raw_vs_filtered(rows: list[dict[str, str]]) -> list[dict]:
    out = []
    for r in rows:
        if r.get('method') != 'star':
            continue
        out.append({
            'task': r['task'],
            'seed': r['seed'],
            'raw_return': f(r, 'raw_return'),
            'filtered_return': f(r, 'filtered_return'),
            'delta_return_filtered_minus_raw': f(r, 'filtered_return') - f(r, 'raw_return'),
            'raw_cost': f(r, 'raw_episode_cost'),
            'filtered_cost': f(r, 'filtered_episode_cost'),
            'delta_cost_filtered_minus_raw': f(r, 'filtered_episode_cost') - f(r, 'raw_episode_cost'),
            'raw_evr': f(r, 'raw_EVR'),
            'filtered_evr': f(r, 'filtered_EVR'),
            'delta_evr_filtered_minus_raw': f(r, 'filtered_EVR') - f(r, 'raw_EVR'),
            'safe_candidate_fraction': f(r, 'safe_candidate_fraction'),
            'fallback_rate': f(r, 'fallback_rate'),
        })
    fields = ['task','seed','raw_return','filtered_return','delta_return_filtered_minus_raw','raw_cost','filtered_cost','delta_cost_filtered_minus_raw','raw_evr','filtered_evr','delta_evr_filtered_minus_raw','safe_candidate_fraction','fallback_rate']
    write_csv(OUT / 'raw_vs_filtered.csv', out, fields)
    return out


def audit_gap_summary(rows: list[dict[str, str]]) -> list[dict]:
    out = []
    for task in sorted({r['task'] for r in rows}):
        for method in ['star_actor', 'star']:
            vals = [r for r in rows if r['task'] == task and r['method'] == method]
            if not vals:
                continue
            p = [f(r, 'pSVR') for r in vals]
            h = [f(r, 'hidden_unsafe_rate') for r in vals]
            anyu = [f(r, 'any_unsafe_shadow_rate') for r in vals]
            out.append({
                'task': task,
                'method': method,
                'seeds': len(vals),
                'pSVR_mean': mean(p),
                'pSVR_sample_std': stdev(p) if len(p) > 1 else 0.0,
                'hidden_unsafe_rate_mean': mean(h),
                'hidden_unsafe_rate_sample_std': stdev(h) if len(h) > 1 else 0.0,
                'any_unsafe_shadow_rate_mean': mean(anyu),
                'any_unsafe_shadow_rate_sample_std': stdev(anyu) if len(anyu) > 1 else 0.0,
            })
    fields = ['task','method','seeds','pSVR_mean','pSVR_sample_std','hidden_unsafe_rate_mean','hidden_unsafe_rate_sample_std','any_unsafe_shadow_rate_mean','any_unsafe_shadow_rate_sample_std']
    write_csv(OUT / 'audit_gap_summary.csv', out, fields)
    return out


def paired_fraction(rows: list[dict], key: str, pred) -> tuple[int, int]:
    good = sum(1 for r in rows if pred(r))
    return good, len(rows)


def paper_gate(raw_cmp: list[dict], rvf: list[dict], audit: list[dict]) -> None:
    claim_a = 'supported' if audit and all(float(r['pSVR_mean']) > 0 for r in audit) else 'weak'
    # Claim B: STAR-Actor cost improves over pointwise on most task-seed pairs; SAC-Lag mixed.
    b_point = [r for r in raw_cmp if r['comparison'] == 'star_actor_vs_pointwise']
    b_lag = [r for r in raw_cmp if r['comparison'] == 'star_actor_vs_sac_lag']
    bp = paired_fraction(b_point, 'cost', lambda r: float(r['delta_cost_star_minus_baseline']) < 0)
    bl = paired_fraction(b_lag, 'cost', lambda r: float(r['delta_cost_star_minus_baseline']) < 0)
    claim_b = 'supported' if bp[0] > bp[1]/2 and bl[0] > bl[1]/2 else ('weak' if bp[0] > bp[1]/2 or bl[0] > bl[1]/2 else 'not supported')
    c = paired_fraction(rvf, 'cost', lambda r: float(r['delta_cost_filtered_minus_raw']) < 0)
    exec_rows = read_csv(EXEC_GRID)
    selected_exec = None
    selected_cfg_path = Path('configs/star_selected_executor.json')
    if selected_cfg_path.exists():
        import json
        cfg = json.loads(selected_cfg_path.read_text())
        selected_exec = (str(int(cfg.get('star_exec_candidates', -1))), str(float(cfg.get('star_exec_margin', -1))))
    exec_selected_rows = []
    if selected_exec:
        for r in exec_rows:
            if r.get('candidates') == selected_exec[0] and str(float(r.get('margin', 'nan'))) == selected_exec[1]:
                exec_selected_rows.append(r)
    exec_improve = sum(1 for r in exec_selected_rows if float(r.get('filtered_evr', 'nan')) < float(r.get('raw_evr', 'nan')))
    claim_c = 'supported' if exec_selected_rows and exec_improve == len(exec_selected_rows) else ('weak' if c[0] > 0 or exec_improve > 0 else 'not supported')
    corridor_rows = read_csv(CORRIDOR)
    claim_d = 'pending' if not corridor_rows else 'supported' if mean([float(r['delta_cost_current_minus_corridor']) for r in corridor_rows]) > 0 else 'weak'
    oracle_rows = read_csv(ORACLE)
    oracle_supported = sum(1 for r in oracle_rows if r.get('supported') == 'True')
    lines = [
        '# STAR Paper Support Gate', '',
        f'Claim A: {claim_a}',
        'Evidence: predicted pSVR is positive for STAR methods. Oracle simulator snapshot diagnostics are unsupported on this wrapper and are not used as evidence.' if oracle_supported == 0 else 'Evidence includes predicted pSVR and oracle diagnostics.',
        '',
        f'Claim B: {claim_b}',
        f'Evidence: STAR-Actor cost lower than pointwise on {bp[0]}/{bp[1]} paired task-seeds; lower than SAC-Lag-local on {bl[0]}/{bl[1]} paired task-seeds.',
        '',
        f'Claim C: {claim_c}',
        f'Evidence: Default Full STAR filtered cost lower than raw on {c[0]}/{c[1]} paired task-seeds. Selected executor grid improved filtered EVR on {exec_improve}/{len(exec_selected_rows)} task-level comparisons with candidates=8 margin=0.0.',
        '',
        f'Claim D: {claim_d}',
        'Evidence: current-only ablation is still pending.' if not corridor_rows else 'Evidence: reports/star_goal/corridor_vs_current_only.csv.',
        '',
        'Do not hide weak or unsupported results; PointGoal1 remains weaker than CarGoal1 for raw actor improvements.',
    ]
    (OUT / 'paper_support_gate.md').write_text('\n'.join(lines) + '\n')


def paper_tables(rows: list[dict[str, str]], audit: list[dict]) -> None:
    # compact markdown tables for current paper drafting
    lines = ['# Paper Tables', '', '## Main Results', '', '| Task | Method | Raw Return | Raw Cost | Filtered Return | Filtered Cost | pSVR |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for task in sorted({r['task'] for r in rows}):
        for method in ['pointwise','sac_lag','star_actor','star']:
            vals = [r for r in rows if r['task'] == task and r['method'] == method]
            if not vals: continue
            def ms(key):
                xs = [f(r, key) for r in vals]
                return f'{mean(xs):.2f} ± {(stdev(xs) if len(xs)>1 else 0.0):.2f}'
            lines.append(f'| {task} | {method} | {ms("raw_return")} | {ms("raw_episode_cost")} | {ms("filtered_return")} | {ms("filtered_episode_cost")} | {ms("pSVR")} |')
    lines += ['', '## Audit Summary', '', '| Task | Method | pSVR | Hidden unsafe rate | Any unsafe shadow rate |', '| --- | --- | ---: | ---: | ---: |']
    for r in audit:
        lines.append(f'| {r["task"]} | {r["method"]} | {float(r["pSVR_mean"]):.3f} ± {float(r["pSVR_sample_std"]):.3f} | {float(r["hidden_unsafe_rate_mean"]):.3f} ± {float(r["hidden_unsafe_rate_sample_std"]):.3f} | {float(r["any_unsafe_shadow_rate_mean"]):.3f} ± {float(r["any_unsafe_shadow_rate_sample_std"]):.3f} |')
    (OUT / 'paper_tables.md').write_text('\n'.join(lines) + '\n')


def paper_claim_text() -> None:
    gate = (OUT / 'paper_support_gate.md').read_text() if (OUT / 'paper_support_gate.md').exists() else ''
    text = ['# Paper Claim Text', '', 'Recommended framing:', '', '- Emphasize counterfactual shadow audit and safe candidate execution rather than claiming uniform raw actor dominance.', '- State clearly that predicted shadow violations are critic-predicted risks, not actual executed violations.', '- Report raw and filtered evaluation separately.', '- Note oracle snapshot diagnostics were attempted but unsupported by the current wrapper chain.', '', 'Current gate:', '', gate]
    (OUT / 'paper_claim_text.md').write_text('\n'.join(text) + '\n')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copy_configs()
    rows = selected_rows()
    raw_cmp = raw_actor_comparison(rows)
    rvf = raw_vs_filtered(rows)
    audit = audit_gap_summary(rows)
    paper_gate(raw_cmp, rvf, audit)
    paper_tables(rows, audit)
    paper_claim_text()
    print(f'wrote reports in {OUT}')

if __name__ == '__main__':
    main()
