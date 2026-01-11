# Main Pipeline - runs full experiment from raw data to analysis
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime


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
    output_dir: str = "runs"
):
    # Run full pipeline from raw data to final analysis
    print_header("FINANCIAL BEHAVIOR ANALYSIS PIPELINE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {output_dir}/")

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(PROJECT_ROOT)
    print(f"Project: {PROJECT_ROOT}")

    RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

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
            cfg = "configs/baseline/regression.yaml" if os.path.exists("configs/baseline/regression.yaml") else "configs/default.yaml"
            run_dir = run_baseline(cfg, dataset_path)
            results['baseline_regression'] = run_dir
            run_dirs.append(run_dir)
            print(f"[OK] {run_dir}")
        except Exception as e:
            print(f"[WARN] {e}")

        print("[1b] Classification...")
        try:
            if os.path.exists("configs/baseline/classification.yaml"):
                run_dir = run_baseline("configs/baseline/classification.yaml", dataset_path)
                results['baseline_classification'] = run_dir
                run_dirs.append(run_dir)
                print(f"[OK] {run_dir}")
        except Exception as e:
            print(f"[WARN] {e}")
    else:
        print_step(1, "BASELINE (SKIP)")

    # STEP 2: Augmentation (Sprint 2)
    if not skip_augmentation:
        print_step(2, "AUGMENTATION (Sprint 2)")
        try:
            from runners.augmentation_experiment import run_augmentation_experiment
            cfg = "configs/augmentation/experiment.yaml"
            if os.path.exists(cfg):
                run_dir, verdict, _ = run_augmentation_experiment(cfg, dataset_path)
                results['augmentation'] = run_dir
                run_dirs.append(run_dir)
                print(f"[OK] {run_dir} - {verdict}")
        except Exception as e:
            print(f"[WARN] {e}")
    else:
        print_step(2, "AUGMENTATION (SKIP)")

    # STEP 3: Multi-Task (Sprint 4)
    if not skip_multitask:
        print_step(3, "MULTI-TASK (Sprint 4)")
        try:
            from runners.run_multitask_experiment import run_multitask_experiment
            cfg = "configs/multitask/experiment.yaml"
            if os.path.exists(cfg):
                run_dir = run_multitask_experiment(cfg, dataset_path)
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
                if os.path.exists(cfg):
                    run_dir, verdict, _ = run_domain_transfer_experiment(cfg, dataset_path, gmsc)
                    results['domain_transfer'] = run_dir
                    run_dirs.append(run_dir)
                    print(f"[OK] {run_dir} - {verdict}")
            except Exception as e:
                print(f"[WARN] {e}")
        else:
            print("[WARN] GMSC not found")
    else:
        print_step(4, "DOMAIN TRANSFER (SKIP)")

    # STEP 5: Analysis (Sprint 6)
    if not skip_analysis and run_dirs:
        print_step(5, "ANALYSIS (Sprint 6)")
        try:
            from runners.run_analysis import run_full_analysis
            analysis_result = run_full_analysis(run_dirs=run_dirs, dataset_path=dataset_path, output_dir="analysis_output")
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
            results['tests'] = "PASSED" if r.returncode == 0 else "FAILED"
            print(f"[OK] {results['tests']}")
        except Exception as e:
            results['tests'] = "ERROR"
            print(f"[WARN] {e}")
    else:
        print_step(6, "TESTS (SKIP)")

    # Summary
    print_header("SUMMARY")
    for k, v in results.items():
        status = "[OK]" if v else "[--]"
        print(f"  {status} {k}: {v or 'Not run'}")
    print("\nOutputs:")
    for d in run_dirs:
        print(f"  - {d}")
    print("\n" + "=" * 70 + "\nPIPELINE COMPLETE!\n" + "=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(description='Run financial analysis pipeline')
    parser.add_argument('--data', '-d', type=str, default=None)
    parser.add_argument('--raw', '-r', type=str, default=None)
    parser.add_argument('--output', '-o', type=str, default='runs')
    parser.add_argument('--skip-preprocessing', action='store_true')
    parser.add_argument('--skip-baseline', action='store_true')
    parser.add_argument('--skip-augmentation', action='store_true')
    parser.add_argument('--skip-multitask', action='store_true')
    parser.add_argument('--skip-transfer', action='store_true')
    parser.add_argument('--skip-analysis', action='store_true')
    parser.add_argument('--skip-tests', action='store_true')
    args = parser.parse_args()

    run_full_pipeline(
        raw_data_path=args.raw, encoded_data_path=args.data,
        skip_preprocessing=args.skip_preprocessing, skip_baseline=args.skip_baseline,
        skip_augmentation=args.skip_augmentation, skip_multitask=args.skip_multitask,
        skip_transfer=args.skip_transfer, skip_analysis=args.skip_analysis,
        skip_tests=args.skip_tests, output_dir=args.output
    )


if __name__ == '__main__':
    main()
