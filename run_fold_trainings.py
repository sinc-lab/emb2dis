"""
Run train_test_model once per fold and repeat each fold multiple times.

Usage example:
    python run_fold_trainings.py

This script uses config/base.yaml + config/env.yaml, then overrides the
fold-specific data_path for each run.
"""

import copy
from datetime import datetime
from pathlib import Path

from src.utils import ConfigLoader
from train_test_model import train_test_model


def get_fold_paths(data_path: str) -> list[Path]:
    """Return all fold directories in the given data_path."""
    configured_path = Path(data_path).expanduser()
    fold_root = configured_path.parent if configured_path.name.startswith("fold_") else configured_path

    fold_paths = sorted(
        path for path in fold_root.iterdir() if path.is_dir() and path.name.startswith("fold_")
    )
    if not fold_paths:
        raise FileNotFoundError(f"No fold directories found under {fold_root}")

    return fold_paths


def main():
    config_loader = ConfigLoader()
    config = config_loader.load()

    fold_paths = get_fold_paths(config["data_path"])
    repeats = int(config.get("repeats", 1))

    output_root = Path(f'results/{config["pLM"]}/5fold/') 
    output_root.mkdir(parents=True, exist_ok=True)


    for fold_dir in fold_paths:
        fold_name = fold_dir.name

        for repeat_idx in range(repeats):
            # Experiment name
            timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            net_param = (
                f"filt{config['filters']}_ker{config['kernel_size']}_resnet{config['n_resnet']}_"
                f"win{config['win_len']}_lr{config['lr']:.0e}"
            )

            exp_identifier = f"{net_param}_{timestamp}"
            run_dir = output_root / fold_name / exp_identifier
            run_dir.mkdir(parents=True, exist_ok=True)

            config = copy.deepcopy(config)
            config["data_path"] = f"{fold_dir.as_posix().rstrip('/')}/"

            config_loader.update(config)
            config_loader.save(run_dir)

            print(f"\n=== Running {exp_identifier} ===")
            train_test_model(config, run_dir)


if __name__ == "__main__":
    main()