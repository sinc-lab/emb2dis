import csv
import yaml
import pandas as pd
import numpy as np
import torch as tr
from torch.nn.functional import softmax
from time import time
from datetime import datetime
from pathlib import Path
from scipy.signal import medfilt
from tabulate import tabulate
from torch.utils.data import DataLoader
from src.dataset import SegmentDataset, AminoAcidDataset


def load_data(
        data_path: str,
        config: dict,
        is_segment: bool = True,
        is_training: bool = True,
        num_workers: int = 1,
        categories: tuple = ('structured', 'disordered'),
    ) -> tuple [DataLoader, int]:
    """
    Loads a dataset for training or evaluation and returns a DataLoder and its length.

    Args:
        data_path: Path to the data source or file.
        config: Configuration dictionary.
        is_segment: Whether to load the segmented dataset (SegmentDataset) or 
                    the amino-acid-level dataset (AminoAcidDataset).
        is_training: Whether this is a training run (enables shuffle and 
                     centered window in segment).
        num_workers: Number of workers for DataLoader.
        categories: Categories to classify sequences, Defaults to ("structured", "disordered"). 

    Returns:
        A tuple containing the DataLoader and the length of the dataset.
    """
    # IMPROVE: change "is_segment" arg name to describe better what it does
    debug = config['debug']
    emb_path_dir = config['emb_path']
    plm = config['pLM']
    emb_path = f"{emb_path_dir}/{plm}/"
    win_len = config['win_len']

    if is_segment:
        dataset = SegmentDataset(data_path, emb_path, categories, win_len,
                                is_training=True, debug=debug)
    else: 
        dataset = AminoAcidDataset(data_path, emb_path, win_len=win_len, 
                                   categories=categories, debug=debug)

    loader = DataLoader(dataset, batch_size=config['batch_size'],
                        shuffle=is_training, num_workers=num_workers, 
                        pin_memory=False) # Disabled to reduce memory usage

    return loader, len(dataset)
    

def calculate_disorder_percentage(predictions, threshold=0.5) -> dict:
    """
    Calculate the percentage of disordered residues.
    
    Args:
        predictions: Tensor of shape (L, 2) with [structured_score, disordered_score]
        threshold: Threshold to classify residues as disordered
    Returns:
        Dictionary with different disorder statistics
    """
    # Get disordered scores
    disorder_scores = predictions[:, 1].numpy()
    
    # Count residues with disorder score > threshold
    disordered_count = np.sum(disorder_scores > threshold)
    total_residues = len(disorder_scores)
    disorder_percentage = (disordered_count / total_residues) * 100
    
    return {
        'disorder_percentage': disorder_percentage,
        'disordered_residues': disordered_count,
        'total_residues': total_residues,
        'disorder_scores': disorder_scores
    }

def predict_sliding_window(net, emb, window_len, step=1, use_softmax=True,
                           median_filter_size=None) -> tuple:
    """
    Predict disorder scores using a sliding window approach.
    
    Args:
        net: Trained neural network model
        emb: Protein embedding tensor (emb_dim, L)
        window_len: Length of sliding window
        step: Step size for sliding window
        use_softmax: Whether to apply softmax to outputs
        median_filter_size: Median filter kernel size for smoothing predictions
    Returns:
        centers: Array of center positions
        predictions: Prediction scores (structured, disordered) for each position
    """
    L = emb.shape[1]
    centers = np.arange(0, L, step)
    batch = tr.zeros((len(centers), emb.shape[0], window_len), dtype=tr.float)

    # Create batch of windows
    for k, center in enumerate(centers):
        start = max(0, center - window_len // 2)
        end = min(L, center + window_len // 2)
        batch[k, :, :end - start] = emb[:, start:end].unsqueeze(0)

    # Predict
    with tr.no_grad():
        pred = net(batch).cpu().detach()
    
    if use_softmax:
        pred = softmax(pred, dim=1)
    
    if median_filter_size is not None and median_filter_size > 0:
        pred = tr.tensor(medfilt(pred.numpy(), kernel_size=(median_filter_size, 1)))

    return centers, pred

def get_embedding_size(plm_name: str) -> int:
    """Returns the embedding size for a given protein language model."""
    plm_sizes = {
        'ProtT5': 1024,
        'ProstT5': 1024,
        'ESM2': 1280,
        'esmc_300m': 960,
        'esmc_600m': 1152
    }
    if plm_name in plm_sizes:
        return plm_sizes[plm_name]
    else:
        raise ValueError(f"Unknown PLM name: {plm_name}")

class ConfigLoader:
    def __init__(self, 
                 model_path: str = 'config/base.yaml', 
                 env_path: str = 'config/env.yaml'):
        """
        Initializes the ConfigLoader class with paths to model and environment 
        configuration files.

        Args:
            model_path: Path to the model config YAML file.
            env_path: Path to the environment config YAML file.
        """
        self.model_path = model_path
        self.env_path = env_path
        
        self.config = None # Combined configuration dictionary
        self.model = None  # Model configuration dictionary

    def load(self) -> dict:
        """
        Loads a model configuration and merges it with environment-specific 
        settings.
        Returns:
            dict: Combined configuration dictionary.
        """
        self.model = self._load_yaml(self.model_path)
        env = self._load_yaml(self.env_path)
        self.config = {**self.model, **env}
        # # add plm to emb_path
        # emb_path = self.config['emb_path']
        # emb_path = Path(emb_path) / self.config.get('plm', 'esm2') 
        # self.config['emb_path'] = f"{str(emb_path)}/"
        return self.config
    
    def save(self, path: str):
        """
        Saves the model dict to a YAML file.
        Args:
            path: Path to save the configuration file.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")
        file = Path(path) / "config.yaml"
        with open(file, 'w') as f:
            yaml.dump(self.model, f, default_flow_style=False)
        print(f"Configuration saved to {file}")

    def update(self, new_config: dict):
        """
        Updates the current configuration with a new configuration dictionary.
        Args:
            new_config (dict): New configuration dictionary to merge with the existing one.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")
        self.config.update(new_config)
        self.model.update(new_config)
        print("Configuration updated.")    

    def get_config(self) -> dict:
        """
        Returns the loaded configuration dictionary.
        Returns:
            dict: The loaded configuration dictionary.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")
        return self.config

    @staticmethod
    def _load_yaml(path: str) -> dict:
        """
        Loads a YAML file from the given path and returns its content as 
        a dictionary
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f)

class ResultsTable():
    """Save results in a DataFrame and export to CSV."""
    def __init__(self):
        """Initializes the ResultsTable"""
        self.metrics = ["auc", "aps", "f1", "fmax", "mcc", 
                        "err", "balanced_acc", "precision", "recall", "threshold"]
        self.df = pd.DataFrame(columns=["Dataset"] + self.metrics)

    def add_entry(self, dataset, **metrics):
        """Add a new entry to the results DataFrame"""
        new_row = {"Dataset": dataset}

        for metric in self.metrics:
            if metric in metrics:
                new_row[metric] = round(metrics[metric], 3)
            else:
                new_row[metric] = float('nan')

        self.df.loc[len(self.df)] = new_row

    def save(self, filepath):
        """Save the results DataFrame to a CSV file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        self.df.to_csv(filepath, index=False)

    def print(self):
        """Print the results DataFrame in a tabular format"""
        print(tabulate(self.df, headers='keys', tablefmt='pretty', showindex=False))

class TimeTracker:
    def __init__(self):
        """
        Initializes the TimeTracker class.
        """
        self.start_time = None
        self.end_time = None
        self.timestamps = []
        
    def start(self):
        self.start_time = time()
        self.timestamps.append({
            "Event": "Start",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Elapsed Time (minutes)": None
        })

    def end(self):
        self.end_time = time()
        elapsed_time = self._execution_time(self.start_time, self.end_time)
        self.timestamps.append({
                    "Event": "End",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Elapsed Time (minutes)": elapsed_time
                })
        print(f'Execution time: {elapsed_time:.2f} minutes', end=' - ')
        
    def step(self, title='Step'):
        step_time = time()
        elapsed_time = self._execution_time(self.start_time, step_time)
        self.timestamps.append({
            "Event": title,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Elapsed Time (minutes)": elapsed_time
        })
        print(f'Execution time: {elapsed_time:.2f} minutes', end=' - ')
        print(f'Ended on {datetime.now().strftime("%a %d %b %Y, %H:%M")}')

    def save_timestamps(self, filename):
        try:
            with open(filename, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["Event", "Timestamp", "Elapsed Time (minutes)"])
                writer.writeheader()  # Write the header row
                writer.writerows(self.timestamps)  # Write all rows from the timestamps list
            print(f"Timestamps saved to {filename}")
        except IOError as e:
            print(f"An error occurred while saving timestamps to {filename}: {e}")

    @staticmethod
    def _execution_time(start, end):
        return (end - start) / 60  # Convert seconds to minutes