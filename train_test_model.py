''' 
IMPORTANT: Run this script from the root directory (not from scripts/)
python -m train_test_model
'''
import os
import sys
import warnings
import torch as tr
import gc  # Add garbage collection
from pathlib import Path
from datetime import datetime
sys.path.append(os.getcwd()) # to correctly import modules from the root directory
from src.train import train
from src.test import test, predict_fasta_to_caid
from src.model import BaseModel
from src.utils import ResultsTable, ConfigLoader, get_embedding_size
from src.metrics import get_caid_metrics

# Configure PyTorch multiprocessing for better memory management
tr.multiprocessing.set_sharing_strategy('file_system')
tr.backends.cudnn.benchmark = False  # Disable cudnn benchmark for consistent memory usage
warnings.filterwarnings("ignore", # Filter some annoying warnings
                        message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")
warnings.simplefilter("ignore", FutureWarning)


def load_base_model(config, base_path):
    """ Load the base model architecture and weights. """
    emb_size = get_embedding_size(config.get('pLM'))

    model = BaseModel(
        len(config['categories']),
        emb_size=emb_size,
        lr=config['lr'],
        device=config['device'],
        filters=config['filters'],
        kernel_size=config['kernel_size'],
        num_layers=config['n_resnet'],
        p_dropout=config['p_dropout'], 
        loss_class_weights=config['loss_class_weights']
        ) 
    model_path = base_path / 'weights.pk'
    model.load_state_dict(tr.load(model_path))

    return model.to(config['device'])

def train_test_model(config, base_path):

    print('TRAINING MODEL')
    train(config, base_path)
    
    # Clear cache and collect garbage to free memory
    tr.cuda.empty_cache() if tr.cuda.is_available() else None
    gc.collect()

    # TESTING MODEL
    print('TESTING MODEL')
    model = load_base_model(config, base_path)
    results_table = ResultsTable()

    datasets = config.get('datasets_to_test', ['dev'])  # Default to 'dev' if not specified
    for dataset in datasets:
        print(f'EVALUATING ON {dataset.upper()} SET')
        metrics = test(
            model,
            config,
            output_path=str(base_path),
            partition=dataset,
            save_predictions=True
            )
        results_table.add_entry(dataset, **metrics)

    # Save results
    results_table.save(base_path / 'results.csv')
    results_table.print()

    prediction_fasta = config.get('prediction_fasta')
    if prediction_fasta:
        print('PREDICTING FASTA INPUT')
        predict_fasta_to_caid(
            model,
            config,
            fasta_path=prediction_fasta,
            output_path=str(base_path),
        )
        get_caid_metrics(base_path, Path(config['caid_path']).parent)

    print('Done :)')

def main():
    # Load the configuration file
    config_loader = ConfigLoader()
    config = config_loader.load()

    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

    # Experiment name
    net_param = (
        f"filt{config['filters']}_ker{config['kernel_size']}_resnet{config['n_resnet']}_"
        f"win{config['win_len']}_lr{config['lr']:.0e}"
    )
    exp_name = f"{net_param}_{timestamp}"
    base_path = Path(f'results/models/{config["pLM"]}/{exp_name}/') 
    base_path.mkdir(parents=True, exist_ok=True)

    # Save the configuration
    config_loader.save(base_path)

    train_test_model(config, base_path)

if __name__ == "__main__":
    main()