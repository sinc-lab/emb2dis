"""
Run an Optuna hyperparameter search for emb2dis.

Hyperparameters and Optuna settings are specified in config/grid_search.yaml. 
Each trial trains and evaluates the model using a sampled hyperparameter configuration,
and the validation AUC is used as the objective to maximize. 

Results are saved in results/grid_search_<timestamp>/, including a CSV of all 
trials and the best trial.

Usage:
    python -m scripts.grid_search

Configuration (config/grid_search.yaml):
    search_space    Hyperparameters and ranges to explore
    study           Optuna study settings, including the number of trials, 
                        startup_trials and optimization objective.
"""
import shutil
import warnings
import optuna
import pandas as pd
import torch as tr
import yaml
from pathlib import Path
from datetime import datetime
from train_test_model import train_test_model
from src.utils import ConfigLoader

tr.multiprocessing.set_sharing_strategy('file_system')
tr.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore",
                        message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")
warnings.simplefilter("ignore", FutureWarning)

GS_CONFIG_PATH = Path("config/grid_search.yaml")
TIMESTAMP = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
RESULTS_DIR = Path(f"results/grid_search_{TIMESTAMP}/")

def load_gs_config() -> dict:
    with open(GS_CONFIG_PATH) as f:
        return yaml.safe_load(f)

def objective(trial):

    # Load configs
    config_loader = ConfigLoader()
    config = config_loader.load()
    gs_config = load_gs_config()
    ss = gs_config['search_space']

    # Suggest parameters from the search space config
    params = {
        'lr':          trial.suggest_float('lr',          ss['lr']['low'],          ss['lr']['high'],          log=ss['lr']['log']),
        'win_len':     trial.suggest_int(  'win_len',     ss['win_len']['low'],     ss['win_len']['high'],     step=ss['win_len']['step']),
        'filters':     trial.suggest_int(  'filters',     ss['filters']['low'],     ss['filters']['high'],     step=ss['filters']['step']),
        'kernel_size': trial.suggest_int(  'kernel_size', ss['kernel_size']['low'], ss['kernel_size']['high'], step=ss['kernel_size']['step']),
        'n_resnet':    trial.suggest_int(  'n_resnet',    ss['n_resnet']['low'],    ss['n_resnet']['high']),
        'p_dropout':   trial.suggest_float('p_dropout',   ss['p_dropout']['low'],   ss['p_dropout']['high'],   step=ss['p_dropout']['step']),
    }

    # Update the config with the trial parameters
    config_loader.update(params)
    config = config_loader.get_config()
    
    # Build a path for this trial
    net_param = (
        f"filt{config['filters']}_ker{config['kernel_size']}_resnet{config['n_resnet']}_"
        f"win{config['win_len']}_lr{config['lr']:.0e}_drop{config['p_dropout']:.2f}"
    )
    current_timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    trial_name = f"trial{trial.number}_{net_param}_{current_timestamp}"
    base_path = RESULTS_DIR / trial_name
    base_path.mkdir(parents=True, exist_ok=True)
    config_loader.save(base_path) 

    # Run training and evaluation
    train_test_model(config, base_path)

    # Read results saved by train_test_model and extract dev AUC
    results_df = pd.read_csv(base_path / 'results.csv')
    auc = results_df.loc[results_df['Dataset'] == 'dev', 'auc'].values[0]

    return auc

def main():

    gs_config = load_gs_config()
    study_cfg = gs_config['study']

    name = f"search_{TIMESTAMP}"

    # Save a copy of the grid search config alongside the results
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(GS_CONFIG_PATH, output_dir / f'grid_search_config_{name}.yaml')

    # Sampler configuration
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=study_cfg['n_startup_trials'],
        multivariate=True      # considers hyperparameter combinations
    )

    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=study_cfg['n_trials'])

    # Save study results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / f'optuna_trials_{name}.csv', index=False)

    best_trial = study.best_trial
    print(f"\n=== Best Trial (AUC) ===")
    print(f"  Trial {best_trial.number}: AUC={best_trial.value:.4f} | Params: {best_trial.params}")

    # Save best trial
    best_df = pd.DataFrame([{
        'trial': best_trial.number,
        'auc': best_trial.value,
        **best_trial.params
    }])

    best_df.to_csv(
        output_dir / f'best_trial_{name}.csv',
        index=False
    )

if __name__ == "__main__":
    main()