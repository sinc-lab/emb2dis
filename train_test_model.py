''' 
IMPORTANT: Run this script from the root directory (not from scripts/)
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
from src.test import test
from src.model import BaseModel
from src.utils import ResultsTable, ConfigLoader

# Configure PyTorch multiprocessing for better memory management
tr.multiprocessing.set_sharing_strategy('file_system')
tr.backends.cudnn.benchmark = False  # Disable cudnn benchmark for consistent memory usage
warnings.filterwarnings("ignore", # Filter some annoying warnings
                        message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")
warnings.simplefilter("ignore", FutureWarning)

def train_test_model(config, base_path):

    print('TRAINING MODEL')
    train(config, base_path)
    
    # Clear cache and collect garbage to free memory
    tr.cuda.empty_cache() if tr.cuda.is_available() else None
    gc.collect()

    # TESTING MODEL
    print('TESTING MODEL')

    categories = config['categories']

    # Load the model
    model_path = base_path / 'weights.pk'
    model = BaseModel(len(categories), emb_size=config['emb_size'][config['pLM']],
                       lr=config['lr'], device='cuda', p_dropout=config['p_dropout'],
                filters=config['filters'], kernel_size=config['kernel_size'],
                num_layers=config['n_resnet']) 
    model.load_state_dict(tr.load(model_path))

    results_table = ResultsTable()

    caid = "caid3_3"  # CAID dataset version
    for partition in ['dev',
                      f"{caid}/disorder_pdb", 
                      f"{caid}/disorder_nox", 
                      f"{caid}/binding",
                      f"{caid}/binding_idr",
                      f"{caid}/linker"]:
                      
        print(f'EVALUATING ON {partition.upper()} SET')

        metrics = test(
            model, config, partition=partition, 
            # save_predictions=True, 
            # output_path=base_path
            )
        results_table.add_entry(partition, **metrics)

    # Save results
    results_table.save(base_path / 'results.csv')
    results_table.print()  # Print the results in a tabular format
    print('Done :)')

if __name__ == "__main__":

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
    base_path = Path(f'results/models/{exp_name}/') 
    base_path.mkdir(parents=True, exist_ok=True)

    # Save the configuration
    config_loader.save(base_path)

    train_test_model(config, base_path)