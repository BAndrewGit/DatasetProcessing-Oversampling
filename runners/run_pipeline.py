# Main Pipeline - runs full experiment from raw data to analysis
import sys
import os

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
        'preprocessing': None, 'baseline_regression': None, 'baseline_classification': None,
        'augmentation': None, 'multitask': None, 'domain_transfer': None, 'analysis': None, 'tests': None
    }
    run_dirs = []

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

    # STEP 2: LATENT SPACE OVERSAMPLING (replaces old jitter/SMOTE augmentation)
    # This is now the primary synthetic data generation method using PCA latent space + clustering
    if not skip_augmentation:
        print_step(2, "LATENT SPACE OVERSAMPLING (Sprint 2 + Sprint 7)")
        print("Using PCA latent space + clustering for synthetic data generation")
        print("(Replaced old jitter/SMOTE methods with proper latent space sampling)")
        try:
            latent_cfg = 'configs/latent_sampling/experiment.yaml'
            latent_test_cfg = 'configs/latent_sampling/test_experiment.yaml'
            cfg_lat = latent_test_cfg if (test_mode and os.path.exists(latent_test_cfg)) else latent_cfg

            if os.path.exists(cfg_lat):
                out_latent = os.path.join(PIPELINE_RUN_DIR, f'latent_oversampling_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                import subprocess
                cmd = [sys.executable, os.path.join(PROJECT_ROOT, 'runners', 'run_latent_sampling_experiment.py'),
                       '--config', cfg_lat, '--dataset', dataset_path, '--output', out_latent]
                print(f"Running latent space oversampling: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)

                if result.returncode == 0:
                    results['augmentation'] = out_latent
                    results['latent'] = out_latent  # Also record as latent for backward compatibility
                    run_dirs.append(out_latent)
                    print(f"[OK] Latent oversampling saved to: {out_latent}")

                    # Determine verdict based on metrics
                    try:
                        import json as _json
                        metrics_path = os.path.join(out_latent, 'metrics.json')
                        if os.path.exists(metrics_path):
                            with open(metrics_path) as mf:
                                metrics = _json.load(mf)
                            synth_used = metrics.get('folds_using_synth', 0)
                            total_folds = metrics.get('total_folds', 1)
                            verdict = 'useful' if synth_used > total_folds * 0.5 else 'not_useful'
                            print(f"    Verdict: {verdict} ({synth_used}/{total_folds} folds used synthetic)")
                    except Exception:
                        pass
                else:
                    print(f"[WARN] Latent oversampling failed (rc={result.returncode})")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:500]}")
            else:
                print("[WARN] No latent config found; skipping latent oversampling")
                print(f"    Expected: {latent_cfg} or {latent_test_cfg}")
        except Exception as e:
            import traceback
            print(f"[WARN] Latent oversampling error: {e}")
            traceback.print_exc()
    else:
        print_step(2, "LATENT SPACE OVERSAMPLING (SKIP)")

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
        gmsc = "data/gmsc/GiveMeSomeCredit-training.csv"
        if os.path.exists(gmsc):
            try:
                from runners.run_domain_transfer_experiment import run_domain_transfer_experiment
                cfg = "configs/transfer/domain_transfer.yaml"
                if test_mode and os.path.exists("configs/transfer/test_domain_transfer.yaml"):
                    cfg = "configs/transfer/test_domain_transfer.yaml"
                if os.path.exists(cfg):
                    run_dir, verdict, _ = run_domain_transfer_experiment(cfg, dataset_path, gmsc, output_dir=PIPELINE_RUN_DIR)
                    results['domain_transfer'] = run_dir
                    run_dirs.append(run_dir)
                    print(f"[OK] {run_dir} - {verdict}")
            except Exception as e:
                print(f"[WARN] {e}")
        else:
            print("[WARN] GMSC not found")
    else:
        print_step(4, "DOMAIN TRANSFER (SKIP)")

    # NOTE: Latent sampling is now integrated into Step 2 (LATENT SPACE OVERSAMPLING)
    # The old Step 4.5 has been removed to avoid duplicate runs

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
                             capture_output=True, text=True, timeout=120, cwd=PROJECT_ROOT)
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
        'run_dirs': run_dirs
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
    args = parser.parse_args()

    run_full_pipeline(raw_data_path=args.raw, encoded_data_path=args.data, test_mode=args.test_mode)


if __name__ == '__main__':
    main()
