import argparse
from pathlib import Path

import pandas as pd

from src.vectorized_metrics.vectorized_metrics import bvaluation


PARTITIONS = ("disorder_nox", "disorder_pdb")
OUTPUT_FOLDER = "caid_metrics"


def run_assessment(reference_file: Path, prediction_file: Path, output_dir: Path) -> None:
    """Run CAID benchmarking for one prediction file."""
    bvaluation(
        reference_file,
        [prediction_file],
        outpath=output_dir,
        dataset=True,
        bootstrap=False, # Disable bootstrap confidence/summary files
        target=False,    # Disable target-specific summary files
        accs_to_read=None,
        summary_only=True,
    )


def collect_partition_metrics(output_dir: Path) -> pd.DataFrame:
    """Collect the dataset-level summary metrics written for each partition."""
    frames = []

    for partition in PARTITIONS:
        for metric_name in ("default", "f1s"):
            metrics_file = output_dir / f"{partition}.analysis.all.dataset.{metric_name}.metrics.csv"
            if not metrics_file.exists():
                raise FileNotFoundError(f"Missing metrics file: {metrics_file}")

            frame = pd.read_csv(metrics_file, index_col=0).reset_index(names="prediction")
            frame.insert(0, "partition", partition)
            frame.insert(1, "metric", metric_name)
            frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def get_caid_metrics(results_dir: Path, caid_dir: Path) -> None:
    """Compute CAID metrics for the expected disorder partitions."""
    results_dir = Path(results_dir)
    caid_dir = Path(caid_dir)
    output_dir = results_dir / OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    for partition in PARTITIONS:
        prediction_file = results_dir / "predictions.caid"
        reference_file = caid_dir / "raw" / f"{partition}.fasta"

        if not prediction_file.exists():
            raise FileNotFoundError(f"Missing prediction file: {prediction_file}")
        if not reference_file.exists():
            raise FileNotFoundError(f"Missing reference file: {reference_file}")

        run_assessment(reference_file, prediction_file, output_dir)

    collect_partition_metrics(output_dir).to_csv(results_dir / "caid_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAID metrics on saved predictions.")
    parser.add_argument("results_dir", type=Path, help="Directory containing predictions_*.caid files")
    parser.add_argument(
        "caid_dir",
        type=Path,
        help="Directory containing the CAID raw reference FASTA files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    get_caid_metrics(args.results_dir, args.caid_dir)


if __name__ == "__main__":
    main()