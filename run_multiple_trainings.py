"""
Script to run train_test_model.py multiple times.

Usage:
    python run_multiple_trainings.py 5
    python run_multiple_trainings.py 10
"""
import argparse
import subprocess
import sys


def run_training(run_number):
    """Run a single training instance."""
    print("\n" + "=" * 80)
    print(f"RUN {run_number}")
    print("=" * 80)
    
    try:
        result = subprocess.run([sys.executable, "-m", "train_test_model"])
        if result.returncode != 0:
            print(f"✗ Run {run_number} failed with return code {result.returncode}")
            return False
        else:
            print(f"✓ Run {run_number} completed successfully")
            return True
    except Exception as e:
        print(f"✗ Error running training: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run train_test_base multiple times'
    )
    parser.add_argument(
        'num_runs',
        type=int,
        help='Number of times to run train_test_base'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RUNNING MULTIPLE TRAININGS")
    print("=" * 80)
    print(f"Number of runs: {args.num_runs}\n")
    
    results = []
    for i in range(args.num_runs):
        success = run_training(i + 1)
        results.append((i + 1, success))
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Successful runs: {successful}/{total}\n")
    
    for run_num, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  Run {run_num}: {status}")
    
    if successful == total:
        print(f"\n✓ All {total} runs completed successfully!")
        return 0
    else:
        print(f"\n✗ {total - successful} run(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())