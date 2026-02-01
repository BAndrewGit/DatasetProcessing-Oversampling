import tempfile
import os
import json
from experiments.verdict import compute_sprint_verdict, save_verdict_and_plot


def _make_fold(perf_rows):
    return {'performance_vs_synth': perf_rows}


def test_verdict_regression_improvement_and_variance():
    # create folds where augmented improves MAE by >1% and reduces variance
    fold1 = _make_fold([
        {'synth_count': 0, 'mae': 0.2, 'accepted': False},
        {'synth_count': 10, 'mae': 0.18, 'accepted': True}
    ])
    fold2 = _make_fold([
        {'synth_count': 0, 'mae': 0.22, 'accepted': False},
        {'synth_count': 10, 'mae': 0.20, 'accepted': True}
    ])
    res = compute_sprint_verdict([fold1, fold2], task='regression')
    assert res['verdict'] == 'useful'
    assert res['rationale']['improves'] or res['rationale']['stabilizes']


def test_verdict_classification_not_useful_if_degrades():
    fold1 = _make_fold([
        {'synth_count': 0, 'macro_f1': 0.9, 'accepted': False},
        {'synth_count': 10, 'macro_f1': 0.85, 'accepted': True}
    ])
    fold2 = _make_fold([
        {'synth_count': 0, 'macro_f1': 0.88, 'accepted': False},
        {'synth_count': 10, 'macro_f1': 0.86, 'accepted': True}
    ])
    res = compute_sprint_verdict([fold1, fold2], task='classification')
    assert res['verdict'] == 'not_useful'


def test_save_verdict_and_plot(tmp_path):
    res = {'verdict': 'useful', 'summary': {'mean_baseline': 0.2, 'mean_best_aug': 0.18}}
    out = str(tmp_path)
    save_verdict_and_plot(res, out, task='regression')
    assert os.path.exists(os.path.join(out, 'verdict.json'))
    assert os.path.exists(os.path.join(out, 'verdict_summary.png'))

