"""
Evaluate a trained model. To run:
    python test_model.py
"""
import os
import sys
import warnings
import torch as tr
from pathlib import Path

sys.path.append(os.getcwd()) # to correctly import modules from the root directory
from src.test import test
from src.model import BaseModel
from src.utils import ResultsTable, ConfigLoader

# Configure PyTorch multiprocessing for better memory management
tr.multiprocessing.set_sharing_strategy('file_system')
tr.backends.cudnn.benchmark = False # Disable cudnn benchmark for consistent memory usage
warnings.filterwarnings( # Filter some annoying warnings
    "ignore",
    message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*",
)
warnings.simplefilter("ignore", FutureWarning)

# CONFIG
MODEL_DIR = "/home/sduarte/emb2dis/results/grid_search_esmc_600m_AUC-APS/best_model"
OUTPUT_DIR = None  # Use None to save results in MODEL_DIR
DEVICE = None  # "cuda", "cuda:0", "cpu", None uses config
SAVE_PREDICTIONS = True
PARTITIONS = ["dev", "caid3_3/disorder_pdb", "caid3_3/disorder_nox"] 


def main():

    model_dir = Path(MODEL_DIR)
    config_path = Path(model_dir / "config.yaml")
    weights_path = Path(model_dir / "weights.pk")

    # Load configuration
    config_loader = ConfigLoader(model_path=str(config_path))
    config = config_loader.load()

    # Resolve device
    device = DEVICE

    # Load the model
    model = BaseModel(
        len(config["categories"]),
        emb_size=config["emb_size"][config["pLM"]],
        lr=config["lr"],
        device=device,
        p_dropout=config["p_dropout"],
        filters=config["filters"],
        kernel_size=config["kernel_size"],
        num_layers=config["n_resnet"],
    )
    model.load_state_dict(tr.load(weights_path, map_location=device))

    # Evaluate
    results_table = ResultsTable()
    partitions = PARTITIONS

    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for partition in partitions:
        print(f"EVALUATING ON {partition.upper()} SET")
        metrics = test(
            model,
            config,
            partition=partition,
            save_predictions=SAVE_PREDICTIONS,
            output_path=str(output_dir) if SAVE_PREDICTIONS else None,
        )
        results_table.add_entry(partition, **metrics)

    # get the threshold for the dev partition and update it to the config file
    dev_threshold = results_table.df.loc[results_table.df["Dataset"] == "dev", "threshold"].values[0]
    print(dev_threshold)
    config_loader.update({"threshold": float(dev_threshold),})
    config_loader.save(config_path.parent)  # Save the updated config with the new threshold

    # Save results
    results_table.save(output_dir / "results_test_model.csv")
    results_table.print()
    print("Done :)")


if __name__ == "__main__":
    main()
