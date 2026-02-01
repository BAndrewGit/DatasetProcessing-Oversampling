import json
import os
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_sprint_verdict(fold_perf: List[Dict[str, Any]], task: str = 'regression') -> Dict[str, Any]:
    """
    Compute sprint-level verdict from a list of per-fold performance dicts.

    Each element of fold_perf is expected to be a dict with key 'performance_vs_synth' containing a list
    of rows produced by `run_latent_fold` for that fold (each row has synth_count, accepted, and metric fields).

    Returns a dict with decision: 'useful' or 'not_useful', and rationale & statistics.
    Rules (strict):
      - primary improvement >= 1% OR variance reduction >= 10%
      - No primary metric degrades >2% on any fold where augmentation was accepted
      - Gates must pass in >=20% of folds (i.e., at least 0.2 fraction)

    primary metric: 'mae' (regression, lower better) or 'macro_f1' (classification, higher better)
    """
    primary = 'mae' if task == 'regression' else 'macro_f1'
    # collect baseline and best-aug per fold
    baselines = []
    bests = []
    folds_accepted = 0
    per_fold_details = []

    for f in fold_perf:
        perf = f.get('performance_vs_synth') or []
        # find baseline entry synth_count==0
        baseline_entry = next((r for r in perf if int(r.get('synth_count', 0)) == 0), None)
        if baseline_entry is None:
            # skip fold
            continue
        baseline_metric = baseline_entry.get(primary)
        # find best accepted augmentation across rows
        # accepted rows have r['accepted'] True
        accepted_rows = [r for r in perf if r.get('accepted')]
        best_aug_metric = None
        best_row = None
        if accepted_rows:
            folds_accepted += 1
            # choose best depending on metric direction
            if task == 'regression':
                # lower better
                # pick min non-null
                vals = [(r, r.get(primary)) for r in accepted_rows if r.get(primary) is not None]
                if vals:
                    best_row, best_aug_metric = min(vals, key=lambda x: x[1])
            else:
                vals = [(r, r.get(primary)) for r in accepted_rows if r.get(primary) is not None]
                if vals:
                    best_row, best_aug_metric = max(vals, key=lambda x: x[1])
        # if no accepted augment or no metric, set best_aug_metric equal to baseline (no change)
        if best_aug_metric is None:
            best_aug_metric = baseline_metric

        baselines.append(float(baseline_metric) if baseline_metric is not None else np.nan)
        bests.append(float(best_aug_metric) if best_aug_metric is not None else np.nan)
        per_fold_details.append({'baseline': baseline_metric, 'best_aug': best_aug_metric, 'accepted': bool(best_row is not None)})

    baselines = np.array(baselines, dtype=float)
    bests = np.array(bests, dtype=float)

    # Drop NaN folds
    valid_mask = ~np.isnan(baselines)
    if valid_mask.sum() == 0:
        raise ValueError('No valid fold baseline metrics found')
    baselines = baselines[valid_mask]
    bests = bests[valid_mask]

    # compute mean improvement
    if task == 'regression':
        # improvement positive if baseline > best (lower MAE)
        raw_diff = baselines - bests
        percent_improvement = (raw_diff / np.where(baselines == 0, np.nan, baselines)) * 100.0
        mean_pct_impr = np.nanmean(percent_improvement) * 1.0
    else:
        raw_diff = bests - baselines
        percent_improvement = (raw_diff / np.where(baselines == 0, np.nan, baselines)) * 100.0
        mean_pct_impr = np.nanmean(percent_improvement) * 1.0

    # variance reduction on primary metric across folds (compare var of baseline vs best)
    var_baseline = float(np.nanvar(baselines, ddof=1)) if baselines.size > 1 else 0.0
    var_best = float(np.nanvar(bests, ddof=1)) if bests.size > 1 else 0.0
    # reduction percent = (var_baseline - var_best)/var_baseline *100
    var_reduction_pct = 0.0
    if var_baseline > 0:
        var_reduction_pct = (var_baseline - var_best) / var_baseline * 100.0

    # no primary metric degrades >2% on any fold where augmentation was accepted
    degrade_violations = []
    for d in per_fold_details:
        if d['accepted']:
            b = d['baseline']
            a = d['best_aug']
            if b is None or a is None:
                continue
            if task == 'regression':
                # percent change (worse if a > b)
                change = (a - b) / b * 100.0 if b != 0 else np.inf
                if change > 2.0:
                    degrade_violations.append({'baseline': b, 'aug': a, 'pct_change': change})
            else:
                # classification: worse if a < b
                change = (b - a) / b * 100.0 if b != 0 else np.inf
                if change > 2.0:
                    degrade_violations.append({'baseline': b, 'aug': a, 'pct_change': change})

    total_folds = len(per_fold_details)
    gates_pass_frac = (folds_accepted / total_folds) if total_folds > 0 else 0.0

    # Verdict logic
    improves = mean_pct_impr >= 1.0
    stabilizes = var_reduction_pct >= 10.0
    enough_gates = gates_pass_frac >= 0.2
    no_degrade = len(degrade_violations) == 0

    decision = ( (improves or stabilizes) and enough_gates and no_degrade )
    verdict = 'useful' if decision else 'not_useful'

    rationale = {
        'mean_pct_improvement': float(mean_pct_impr),
        'var_reduction_pct': float(var_reduction_pct),
        'gates_pass_frac': float(gates_pass_frac),
        'degrade_violations_count': len(degrade_violations),
        'improves': bool(improves),
        'stabilizes': bool(stabilizes),
        'enough_gates': bool(enough_gates),
        'no_degrade': bool(no_degrade)
    }

    result = {
        'verdict': verdict,
        'decision': bool(decision),
        'rationale': rationale,
        'per_fold_details': per_fold_details,
        'summary': {
            'mean_baseline': float(np.nanmean(baselines)),
            'mean_best_aug': float(np.nanmean(bests)),
            'var_baseline': var_baseline,
            'var_best': var_best
        }
    }

    return result


def save_verdict_and_plot(verdict: Dict[str, Any], out_dir: str, task: str = 'regression') -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'verdict.json'), 'w') as f:
        json.dump(verdict, f, indent=2)

    # plot baseline vs aug means
    mean_baseline = verdict['summary']['mean_baseline']
    mean_best = verdict['summary']['mean_best_aug']
    plt.figure(figsize=(4,3))
    if task == 'regression':
        plt.bar(['baseline','augmented'], [mean_baseline, mean_best], color=['#777777','#2ca02c'])
        plt.ylabel('MAE (lower better)')
    else:
        plt.bar(['baseline','augmented'], [mean_baseline, mean_best], color=['#777777','#2ca02c'])
        plt.ylabel('Macro-F1 (higher better)')
    plt.title(f'Pool-level summary: verdict={verdict.get("verdict")}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'verdict_summary.png'))
    plt.close()

