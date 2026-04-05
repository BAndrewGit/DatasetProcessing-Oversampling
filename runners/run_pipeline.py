# Main Pipeline - runs full experiment from raw data to analysis
import sys
import os

# Add runners dir to path and import startup to silence warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import _startup  # noqa: F401
except ImportError:
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 4))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
import json

# NOTE: We removed aggressive runtime suppression of sklearn/UserWarning and FutureWarning
# to ensure we surface issues. Code now ensures DataFrame inputs for sklearn estimators.

def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_step(step_num, description):
    print(f"\n[STEP {step_num}] {description}")
    print("-" * 50)


def run_full_pipeline(
    raw_data_path: str = None,
    encoded_data_path: str = None,
    skip_preprocessing: bool = False,
    skip_baseline: bool = False,
    skip_augmentation: bool = False,
    skip_multitask: bool = False,
    skip_transfer: bool = False,
    skip_hybrid: bool = False,
    skip_analysis: bool = False,
    skip_tests: bool = False,
    output_dir: str = "runs",
    test_mode: bool = True
):
    # Create unique run folder for this pipeline execution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_run_name = f"pipeline_{timestamp}"

    print_header("FINANCIAL BEHAVIOR ANALYSIS PIPELINE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run ID: {pipeline_run_name}")

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(PROJECT_ROOT)
    print(f"Project: {PROJECT_ROOT}")

    # Create pipeline run directory - all results go here
    PIPELINE_RUN_DIR = os.path.join(PROJECT_ROOT, output_dir, pipeline_run_name)
    os.makedirs(PIPELINE_RUN_DIR, exist_ok=True)
    print(f"Output: {PIPELINE_RUN_DIR}")

    # Silence joblib/loky physical core warning by setting max CPU count if not present
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 1))

    RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "data", "experiments")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    print(f"\nSearching data:")
    print(f"  Raw: {RAW_DIR}")
    print(f"  Processed: {PROCESSED_DIR}")

    # List raw files
    if os.path.exists(RAW_DIR):
        try:
            csv_in_raw = [f for f in os.listdir(RAW_DIR) if f.lower().endswith('.csv')]
            if csv_in_raw:
                print(f"  CSV files: {csv_in_raw}")
        except Exception as e:
            print(f"  Error: {e}")

    processed_candidates = [
        os.path.join(PROCESSED_DIR, "1_encoded.csv"),
        os.path.join(PROCESSED_DIR, "encoded_dataset.csv"),
    ]
    raw_candidates = [
        os.path.join(RAW_DIR, "FinancialSurvey.csv"),
        os.path.join(RAW_DIR, "survey.csv"),
        os.path.join(RAW_DIR, "raw_data.csv"),
    ]

    dataset_path = None
    needs_preprocessing = False

    # 1. Check processed data
    if encoded_data_path and os.path.exists(encoded_data_path):
        dataset_path = encoded_data_path
        print(f"[OK] Using: {dataset_path}")
    else:
        for p in processed_candidates:
            if os.path.exists(p):
                dataset_path = p
                print(f"[OK] Found: {dataset_path}")
                break

    # 2. Check raw data
    if not dataset_path:
        if raw_data_path and os.path.exists(raw_data_path):
            raw_path = raw_data_path
            needs_preprocessing = True
        else:
            raw_path = None
            from pathlib import Path
            raw_dir_path = Path(RAW_DIR)
            if raw_dir_path.exists():
                csv_files = list(raw_dir_path.glob("*.csv"))
                if csv_files:
                    preferred = [f for f in csv_files if 'survey' in f.name.lower() or 'financial' in f.name.lower()]
                    raw_path = str(preferred[0]) if preferred else str(csv_files[0])
            if not raw_path:
                for p in raw_candidates:
                    if os.path.exists(p):
                        raw_path = p
                        break
            if raw_path:
                needs_preprocessing = True
                print(f"[OK] Raw: {raw_path}")

        # 3. Preprocess
        if needs_preprocessing and raw_path:
            print_step("0", "PREPROCESSING")
            try:
                from FirstProcessing.main import process_dataset
                dataset_path = process_dataset(input_path=raw_path, output_dir=PROCESSED_DIR, output_prefix="1")
                if dataset_path:
                    print(f"[OK] Done: {dataset_path}")
                else:
                    print("[FAIL] Preprocessing failed")
                    return None
            except Exception as e:
                print(f"[FAIL] Error: {e}")
                return None

    if not dataset_path:
        print("\n[ERROR] No dataset found!")
        return None

    print(f"\n>>> Dataset: {dataset_path}")

    results = {
        'preprocessing': None, 'eda': None, 'baseline_regression': None, 'baseline_classification': None,
        'augmentation': None, 'multitask': None, 'domain_transfer': None, 'hybrid': None,
        'final_selection': None, 'analysis': None, 'tests': None
    }
    rejected_experiments = []
    run_dirs = []

    # STEP 0: EDA (Exploratory Data Analysis)
    print_step(0, "EXPLORATORY DATA ANALYSIS")
    try:
        from runners.run_eda import run_eda
        eda_dir = os.path.join(PIPELINE_RUN_DIR, 'eda')
        summary = run_eda(dataset_path, eda_dir, target_col='Risk_Score', task='regression')
        results['eda'] = eda_dir
        print(f"[OK] EDA plots saved to: {eda_dir}")
        print(f"    Generated {len(summary.get('plots_generated', []))} plots")
    except Exception as e:
        import traceback
        print(f"[WARN] EDA failed: {e}")
        traceback.print_exc()

    # STEP 1: Baseline (Sprint 1)
    if not skip_baseline:
        print_step(1, "BASELINE (Sprint 1)")
        from runners.run_baseline import run_baseline

        print("[1a] Regression...")
        try:
            if test_mode and os.path.exists("configs/baseline/test_regression.yaml"):
                cfg = "configs/baseline/test_regression.yaml"
            else:
                cfg = "configs/baseline/regression.yaml" if os.path.exists("configs/baseline/regression.yaml") else "configs/default.yaml"
            run_dir = run_baseline(cfg, dataset_path, output_dir=PIPELINE_RUN_DIR)
            results['baseline_regression'] = run_dir
            run_dirs.append(run_dir)
            print(f"[OK] {run_dir}")
        except Exception as e:
            print(f"[WARN] {e}")

        print("[1b] Classification...")
        try:
            if test_mode and os.path.exists("configs/baseline/test_classification.yaml"):
                cfg = "configs/baseline/test_classification.yaml"
            elif os.path.exists("configs/baseline/classification.yaml"):
                cfg = "configs/baseline/classification.yaml"
            else:
                cfg = "configs/default.yaml"
            run_dir = run_baseline(cfg, dataset_path, output_dir=PIPELINE_RUN_DIR)
            results['baseline_classification'] = run_dir
            run_dirs.append(run_dir)
            print(f"[OK] {run_dir}")
        except Exception as e:
            print(f"[WARN] {e}")
    else:
        print_step(1, "BASELINE (SKIP)")

    # STEP 2: LATENT OVERSAMPLING (REJECTED EXPERIMENT)
    print_step(2, "LATENT OVERSAMPLING (REJECTED)")
    rejection_info = {
        'experiment': 'latent_oversampling',
        'status': 'rejected',
        'used_in_training': False,
        'used_in_reporting': False,
        'reason': 'Compromised by leakage risk and non-publicable metrics; kept only as rejected history.',
    }
    rejected_experiments.append(rejection_info)
    results['augmentation'] = rejection_info
    print("[INFO] Latent oversampling is excluded from final pipeline by policy.")

    # STEP 3: Multi-TASK (Sprint 4)
    if not skip_multitask:
        print_step(3, "MULTI-TASK (Sprint 4)")
        try:
            from runners.run_multitask_experiment import run_multitask_experiment
            cfg = "configs/multitask/experiment.yaml"
            if test_mode and os.path.exists("configs/multitask/test_experiment.yaml"):
                cfg = "configs/multitask/test_experiment.yaml"
            if os.path.exists(cfg):
                run_dir = run_multitask_experiment(cfg, dataset_path, output_dir=PIPELINE_RUN_DIR)
                if run_dir:
                    results['multitask'] = run_dir
                    run_dirs.append(run_dir)
                    print(f"[OK] {run_dir}")
                else:
                    print("[WARN] Multitask returned None")
        except Exception as e:
            import traceback
            print(f"[WARN] Multitask error: {e}")
            traceback.print_exc()
    else:
        print_step(3, "MULTI-TASK (SKIP)")

    # STEP 4: Domain Transfer (Sprint 5)
    if not skip_transfer:
        print_step(4, "DOMAIN TRANSFER (Sprint 5)")

        # Search for GMSC file in multiple locations
        gmsc_candidates = [
            "data/gmsc/GiveMeSomeCredit-training.csv",
            "data/gmsc/cs-training.csv",
            os.path.join(PROJECT_ROOT, "data", "gmsc", "GiveMeSomeCredit-training.csv"),
            os.path.join(PROJECT_ROOT, "data", "gmsc", "cs-training.csv"),
        ]
        gmsc = None
        for candidate in gmsc_candidates:
            if os.path.exists(candidate):
                gmsc = candidate
                print(f"  Found GMSC: {gmsc}")
                break

        if gmsc:
            try:
                from runners.run_domain_transfer_experiment import run_domain_transfer_experiment
                cfg = "configs/transfer/domain_transfer.yaml"
                if test_mode and os.path.exists("configs/transfer/test_domain_transfer.yaml"):
                    cfg = "configs/transfer/test_domain_transfer.yaml"

                if os.path.exists(cfg):
                    print(f"  Using config: {cfg}")
                    print(f"  ADV dataset: {dataset_path}")
                    print(f"  GMSC dataset: {gmsc}")

                    result = run_domain_transfer_experiment(cfg, dataset_path, gmsc, output_dir=PIPELINE_RUN_DIR)

                    # Handle different return types robustly
                    if result is None:
                        print("[WARN] Domain transfer returned None")
                    elif isinstance(result, tuple) and len(result) >= 2:
                        run_dir, verdict = result[0], result[1]
                        if run_dir and os.path.isdir(run_dir):
                            results['domain_transfer'] = run_dir
                            run_dirs.append(run_dir)
                            print(f"[OK] {run_dir}")
                            print(f"    Verdict: {verdict}")
                        else:
                            print(f"[WARN] Domain transfer dir invalid or doesn't exist: {run_dir}")
                    elif isinstance(result, str) and os.path.isdir(result):
                        results['domain_transfer'] = result
                        run_dirs.append(result)
                        print(f"[OK] {result}")
                    else:
                        print(f"[WARN] Domain transfer unexpected result type: {type(result)}")
                else:
                    print(f"[WARN] Config not found: {cfg}")
                    # List available configs for debugging
                    cfg_dir = os.path.join(PROJECT_ROOT, "configs", "transfer")
                    if os.path.exists(cfg_dir):
                        print(f"  Available configs: {os.listdir(cfg_dir)}")
            except Exception as e:
                import traceback
                print(f"[WARN] Domain transfer error: {e}")
                traceback.print_exc()
        else:
            print("[WARN] GMSC dataset not found. Searched:")
            for c in gmsc_candidates:
                print(f"    - {c}")
            print("  Download from: https://www.kaggle.com/c/GiveMeSomeCredit/data")
    else:
        print_step(4, "DOMAIN TRANSFER (SKIP)")

    # NOTE: Latent oversampling is explicitly retired from production pipeline flow.

    # STEP 4.5: Hybrid Multitask + Transfer (Production candidate)
    if not skip_hybrid:
        print_step("4.5", "HYBRID MULTITASK + TRANSFER")
        try:
            from runners.run_hybrid_production import run_hybrid_production
            hybrid_cfg = "configs/transfer/hybrid_transfer.yaml"
            multitask_cfg = "configs/multitask/experiment.yaml"
            gmsc = None
            for candidate in [
                os.path.join(PROJECT_ROOT, "data", "gmsc", "GiveMeSomeCredit-training.csv"),
                os.path.join(PROJECT_ROOT, "data", "gmsc", "cs-training.csv"),
            ]:
                if os.path.exists(candidate):
                    gmsc = candidate
                    break
            if os.path.exists(hybrid_cfg) and os.path.exists(multitask_cfg) and gmsc:
                hybrid_result = run_hybrid_production(
                    multitask_config=multitask_cfg,
                    transfer_config=hybrid_cfg,
                    dataset_path=dataset_path,
                    gmsc_path=gmsc,
                    output_dir=PIPELINE_RUN_DIR,
                    source_pipeline_run=PIPELINE_RUN_DIR,
                    pipeline_run_id=pipeline_run_name,
                )
                results['hybrid'] = hybrid_result
                results['final_selection'] = {
                    'final_model_family': hybrid_result.get('final_model_family'),
                    'decision_dir': hybrid_result.get('decision_dir'),
                    'winner_checkpoint_path': hybrid_result.get('winner_checkpoint_path'),
                    'bundle_dir': hybrid_result.get('bundle', {}).get('bundle_dir'),
                }
                bundle_dir = hybrid_result.get('bundle', {}).get('bundle_dir')
                if bundle_dir:
                    run_dirs.append(bundle_dir)
                print("[OK] Hybrid production bundle generated")
            else:
                print("[WARN] Hybrid step skipped (missing config or GMSC dataset)")
        except Exception as e:
            import traceback
            print(f"[WARN] Hybrid step error: {e}")
            traceback.print_exc()
    else:
        print_step("4.5", "HYBRID MULTITASK + TRANSFER (SKIP)")

    # STEP 5: Analysis (Sprint 6)
    if not skip_analysis and run_dirs:
        print_step(5, "ANALYSIS (Sprint 6)")
        try:
            from runners.run_analysis import run_full_analysis
            analysis_output = os.path.join(PIPELINE_RUN_DIR, "analysis")
            analysis_result = run_full_analysis(run_dirs=run_dirs, dataset_path=dataset_path, output_dir=analysis_output)
            if analysis_result:
                analysis_dir = analysis_result[0] if isinstance(analysis_result, tuple) else analysis_result
                results['analysis'] = analysis_dir
                print(f"[OK] {analysis_dir}")
            else:
                print("[WARN] Analysis returned None")
        except Exception as e:
            import traceback
            print(f"[WARN] Analysis error: {e}")
            traceback.print_exc()
    else:
        print_step(5, "ANALYSIS (SKIP)")

    # STEP 6: Tests
    if not skip_tests:
        print_step(6, "TESTS")
        try:
            import subprocess
            r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
                             capture_output=True, text=True, timeout=120, cwd=PROJECT_ROOT,
                             encoding='utf-8', errors='replace')
            results['tests'] = "PASSED" if r.returncode == 0 else f"FAILED (rc={r.returncode})"
            # Save pytest output for debugging
            try:
                with open(os.path.join(PIPELINE_RUN_DIR, 'pytest_output.txt'), 'w', encoding='utf-8') as pf:
                    pf.write('STDOUT:\n')
                    pf.write(r.stdout or '')
                    pf.write('\nSTDERR:\n')
                    pf.write(r.stderr or '')
            except Exception:
                pass
        except Exception as e:
             print(f"[WARN] Tests error: {e}")
             results['tests'] = "ERROR"

    # Save pipeline summary and exit
    pipeline_summary = {
        'run_id': pipeline_run_name,
        'timestamp': timestamp,
        'results': results,
        'run_dirs': run_dirs,
        'rejected_experiments': rejected_experiments,
    }
    try:
        with open(os.path.join(PIPELINE_RUN_DIR, 'pipeline_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(pipeline_summary, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write pipeline_summary.json: {e}")

    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    for k, v in results.items():
        print(f"  - {k}: {v}")

    print("\nOutputs:")
    for d in run_dirs:
        print(f"  - {d}")

    print("\nPIPELINE COMPLETE!")

    return PIPELINE_RUN_DIR, results


def main():
    parser = argparse.ArgumentParser(description='Run full pipeline')
    parser.add_argument('--raw', type=str, default=None, help='Raw dataset path')
    parser.add_argument('--data', '-d', type=str, default=None, help='Encoded dataset path')
    parser.add_argument('--test-mode', action='store_true', help='Run in fast test mode')
    parser.add_argument('--skip-hybrid', action='store_true', help='Skip hybrid multitask+transfer production step')
    args = parser.parse_args()

    run_full_pipeline(
        raw_data_path=args.raw,
        encoded_data_path=args.data,
        test_mode=args.test_mode,
        skip_hybrid=args.skip_hybrid,
    )


if __name__ == '__main__':
    main()
