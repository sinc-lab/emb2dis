import os
import sys
import warnings
import torch as tr
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, average_precision_score, balanced_accuracy_score, precision_recall_curve
sys.path.append(os.getcwd()) # to correctly import modules
from src.caid_output import load_fasta_sequences, write_caid_block
from src.utils import load_data, predict_sliding_window
tr.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore", # Filter warnings
                        message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")
warnings.simplefilter("ignore", FutureWarning)


def test(
        model: tr.nn.Module,
        config: dict,  
        output_path: str = None,
        partition: str = 'dev', 
        save_predictions: bool = False,  
        ) -> None:
    """
    Evaluate a given model on a test dataset and compute metrics.
    Args:
        model: The model to be evaluated.
        config: Configuration dictionary containing dataset and other parameters.
        output_path: Path to save the predictions and results.
        partition: Partition to evaluate, e.g., 'test'.
        save_predictions: Whether to save predictions to a CSV file. 
    Returns:
        dict: A dictionary containing the evaluation metrics
    """
    use_softmax = config.get('soft_max', True)
    
    # Load the test dataset
    # Use caid_path for caid datasets, otherwise use data_path
    if partition == 'dev':
        dataset_file = str(Path(config['data_path']) / f"{partition}.csv")
    elif 'disorder' in partition.lower():
        dataset_file = str(Path(config['caid_path']) / f"{partition}.csv")
    else:
        raise ValueError(f"Unknown partition: {partition}")

    test_loader, len_test = load_data(dataset_file, config, is_segment=False, 
                                      is_training=False, num_workers=0)
    # Set num_workers=0 to reduce memory problems
    
    # Evaluate the model
    model.eval()
    loss, err, auc, f1, pred, ref_soft, ref_hard, names, centers = model.pred(test_loader)
    
    if use_softmax:
        pred = tr.softmax(pred, dim=1)
    pred_bin = tr.argmax(pred, dim=1).cpu().detach().numpy()
    pred_labels = pred_bin.tolist()

    # Save predictions and references
    if save_predictions:
        output_dir = Path(output_path) if output_path else Path.cwd()
        partition_name = partition.replace('/', '_')
        pred_df = pd.DataFrame({
            'acc': names,
            'centers': centers,
            'structured_score': pred[:, 0],
            'disordered_score': pred[:, 1],
            'reference_label': ref_hard.cpu().detach().numpy(),
            'predicted_label': pred_labels,
        })
        pred_df.to_csv(output_dir / f"predictions_{partition_name}.csv", index=False)


    # Calculate metrics using sklearn
    aps = average_precision_score(ref_hard, pred[:, 1], average='macro')
    recall = recall_score(ref_hard, pred_bin, average='macro', zero_division=0)
    precision = precision_score(ref_hard, pred_bin, average='macro', zero_division=0)
    mcc = matthews_corrcoef(ref_hard, pred_bin)
    balanced_acc = balanced_accuracy_score(ref_hard, pred_bin)

    # F-max: maximum F1 across all thresholds
    p_curve, r_curve, thresholds = precision_recall_curve(ref_hard, pred[:, 1])
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-8)
    fmax = float(f1_curve.max())

    best_index = f1_curve.argmax()
    best_threshold = thresholds[best_index]

    results = {
        'auc': auc,
        'aps': aps,
        'fmax': fmax,
        'f1': f1,
        'mcc': mcc,
        'err': err,
        'balanced_acc': balanced_acc,
        'precision': precision,
        'recall': recall,
        'threshold': best_threshold,
    }

    return results


def predict_fasta_to_caid(
        model: tr.nn.Module,
        config: dict,
        fasta_path: str,
        output_path: str = None,
    ) -> Path:
    """
    Predict disorder for a FASTA file using full-sequence sliding windows 
    and save one file in CAID-format.
    """
    if not fasta_path:
        raise ValueError("fasta_path is required")

    emb_dir = Path(config['emb_path']) / config['pLM']
    output_dir = Path(output_path) if output_path else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = Path(fasta_path)
    sequences = load_fasta_sequences(fasta_path)

    model.eval()
    output_file = output_dir / "predictions.caid"

    # TODO: add progress bar
    with open(output_file, 'w') as handle: 
        for protein_id, sequence in sequences.items():
            emb_file = emb_dir / f"{protein_id}.npy"
            if not emb_file.exists():
                raise FileNotFoundError(f"Missing embedding file: {emb_file}")

            emb = tr.tensor(np.load(emb_file), dtype=tr.float32)
            centers, predictions = predict_sliding_window(
                model,
                emb,
                window_len=config['win_len'],
                step=1,
                use_softmax=config['soft_max'],
                median_filter_size=None,
            )

            predicted_labels = tr.argmax(predictions, dim=1).cpu().tolist()
            rows = [
                (int(center) + 1, sequence[int(center)], float(score), int(label))
                for center, score, label in zip(centers, predictions[:, 1].cpu().tolist(), predicted_labels)
            ]
            write_caid_block(handle, protein_id, rows)

    return output_file