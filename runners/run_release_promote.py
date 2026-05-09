# Promote selected final decision bundle to a stable deployment release and generate plots.

import argparse
import json
import os
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runners.verify_release import verify_release_artifacts


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(path: str, root: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def _latest_final_decision(runs_dir: str) -> str:
    candidates = []
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name)
        if os.path.isdir(path) and name.startswith("final_decision_"):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No final_decision_* directory found in: {runs_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _generate_metric_bar_plot(final_comparison: dict, out_path: str) -> None:
    metrics = final_comparison.get("metrics", {})
    keys = ["risk_mae", "risk_spearman", "savings_macro_f1"]

    mt_vals = []
    hy_vals = []
    labels = []
    for k in keys:
        m = metrics.get(k, {})
        mt = m.get("multitask", {}).get("mean")
        hy = m.get("hybrid", {}).get("mean")
        if mt is None or hy is None:
            continue
        labels.append(k)
        mt_vals.append(mt)
        hy_vals.append(hy)

    if not labels:
        return

    x = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar([i - width / 2 for i in x], mt_vals, width=width, label="multitask")
    plt.bar([i + width / 2 for i in x], hy_vals, width=width, label="hybrid")
    plt.xticks(x, labels, rotation=15)
    plt.title("Final Decision Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _generate_fold_boxplot(final_comparison: dict, out_path: str) -> None:
    metrics = final_comparison.get("metrics", {})
    keys = ["risk_mae", "risk_spearman", "savings_macro_f1"]

    data = []
    labels = []
    for k in keys:
        m = metrics.get(k, {})
        mt = m.get("multitask", {}).get("all")
        hy = m.get("hybrid", {}).get("all")
        if not mt or not hy:
            continue
        data.append(mt)
        labels.append(f"{k}_mt")
        data.append(hy)
        labels.append(f"{k}_hy")

    if not data:
        return

    plt.figure(figsize=(10, 4))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.title("Fold-Level Distribution (Multitask vs Hybrid)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _generate_selection_plot(final_comparison: dict, out_path: str) -> None:
    report = final_comparison.get("selection_report", {})
    checks = ["spearman_ok", "mae_ok", "risk_r2_ok", "stability_ok", "savings_ok"]
    vals = [1 if bool(report.get(k, False)) else 0 for k in checks]

    plt.figure(figsize=(7, 3))
    colors = ["#2ca02c" if v == 1 else "#d62728" for v in vals]
    plt.bar(checks, vals, color=colors)
    plt.ylim(-0.05, 1.05)
    plt.title("Selection Criteria Pass/Fail")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _copy_tree(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def promote_release_bundle(
    decision_dir: str = None,
    runs_dir: str = "runs",
    releases_dir: str = "deployment/releases",
    release_name: str = None,
    update_current: bool = True,
) -> dict:
    root = _project_root()
    runs_dir = _resolve_path(runs_dir, root)
    releases_dir = _resolve_path(releases_dir, root)
    if decision_dir:
        decision_dir = _resolve_path(decision_dir, root)

    decision_dir = decision_dir or _latest_final_decision(runs_dir)
    final_cmp_path = os.path.join(decision_dir, "final_comparison.json")
    comparison_report_path = os.path.join(decision_dir, "comparison_report.json")

    if not os.path.isfile(final_cmp_path):
        raise FileNotFoundError(f"Missing final_comparison.json in {decision_dir}")

    final_comparison = _load_json(final_cmp_path)
    if os.path.isfile(comparison_report_path):
        comparison_report = _load_json(comparison_report_path)
        bundle_src = comparison_report.get("bundle", {}).get("bundle_dir")
    else:
        comparison_report = {}
        bundle_src = None

    if not bundle_src:
        bundle_src = os.path.join(decision_dir, "inference_bundle")
    else:
        bundle_src = _resolve_path(bundle_src, root)

    if not os.path.isdir(bundle_src):
        raise FileNotFoundError(f"Inference bundle not found: {bundle_src}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    release_name = release_name or f"release_{timestamp}"
    release_dir = os.path.join(releases_dir, release_name)
    bundle_dst = os.path.join(release_dir, "bundle")
    plots_dir = os.path.join(release_dir, "plots")
    snapshot_dir = os.path.join(release_dir, "decision_snapshot")

    os.makedirs(release_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    _copy_tree(bundle_src, bundle_dst)

    for name in [
        "final_multitask_results.json",
        "final_hybrid_results.json",
        "final_comparison.json",
        "final_multitask_grad_logs.json",
        "final_hybrid_grad_logs.json",
        "comparison_report.json",
    ]:
        src = os.path.join(decision_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(snapshot_dir, name))

    _generate_metric_bar_plot(final_comparison, os.path.join(plots_dir, "metric_bars.png"))
    _generate_fold_boxplot(final_comparison, os.path.join(plots_dir, "fold_boxplots.png"))
    _generate_selection_plot(final_comparison, os.path.join(plots_dir, "selection_checks.png"))

    manifest = {
        "promoted_at": datetime.now().isoformat(),
        "source_decision_dir": decision_dir,
        "source_bundle_dir": bundle_src,
        "final_model_family": final_comparison.get("final_model_family"),
        "release_dir": release_dir,
        "bundle_dir": bundle_dst,
        "plots_dir": plots_dir,
    }
    _save_json(os.path.join(release_dir, "promotion_manifest.json"), manifest)

    verification = verify_release_artifacts(release_dir=release_dir, releases_dir=releases_dir)
    if not verification.get("ok"):
        raise ValueError(
            "Promoted release failed validation: "
            + "; ".join(str(error) for error in verification.get("errors", []))
        )

    current_dir = os.path.join(root, "deployment", "current")
    if update_current:
        if os.path.isdir(current_dir):
            shutil.rmtree(current_dir)
        _copy_tree(release_dir, current_dir)
        current_manifest = dict(manifest)
        current_manifest["release_dir"] = current_dir
        current_manifest["bundle_dir"] = os.path.join(current_dir, "bundle")
        current_manifest["plots_dir"] = os.path.join(current_dir, "plots")
        current_manifest["current_dir"] = current_dir
        _save_json(os.path.join(current_dir, "promotion_manifest.json"), current_manifest)
        manifest["current_dir"] = current_dir

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote final decision bundle to deployment release with plots")
    parser.add_argument("--decision-dir", default=None, help="Path to final_decision_* directory")
    parser.add_argument("--runs-dir", default="runs", help="Runs root directory used when decision dir is not provided")
    parser.add_argument("--releases-dir", default="deployment/releases", help="Destination releases directory")
    parser.add_argument("--release-name", default=None, help="Optional release name")
    parser.add_argument("--no-current", action="store_true", help="Do not update deployment/current")
    args = parser.parse_args()

    manifest = promote_release_bundle(
        decision_dir=args.decision_dir,
        runs_dir=args.runs_dir,
        releases_dir=args.releases_dir,
        release_name=args.release_name,
        update_current=not args.no_current,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()




