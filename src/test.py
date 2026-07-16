import os
import sys
import warnings
import torch as tr
import pandas as pd
from pathlib import Path
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, average_precision_score, balanced_accuracy_score, precision_recall_curve
sys.path.append(os.getcwd()) # to correctly import modules
from src.utils import load_data
tr.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore", # Filter warnings
                        message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")
warnings.simplefilter("ignore", FutureWarning)


def test(
        model: tr.nn.Module,
        config: dict,  
        output_path: str = None,
        partition: str = 'test', 
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
    use_softmax = config.get('soft_max', False)
    
    # Load the test dataset
    # Use caid_path for caid datasets, otherwise use data_path
    if 'disorder' in partition.lower():
        dataset_file = str(Path(config['caid_path']) / f"{partition}.csv")
    elif partition == "test":
        path = Path(config['data_path'])
        dataset_file = path.parent / "test.csv"
    else:
        path = Path(config['data_path']) / f"{partition}"
        dataset_file = f"{path}.csv"
    
    test_loader, len_test = load_data(dataset_file, config, is_segment=False, 
                                      is_training=False, num_workers=0)
    # Set num_workers=0 to reduce memory problems
    
    # Evaluate the model
    model.eval()
    loss, err, auc, f1, pred, ref_soft, ref_hard, names, centers = model.pred(test_loader)
    
    if use_softmax:
        pred = tr.softmax(pred, dim=1)
    pred_bin = tr.argmax(pred, dim=1).cpu().detach().numpy()

    # Save predictions and references
    if save_predictions:
        pred_df = pd.DataFrame({
            'acc': names,
            'centers': centers,
            'structured_score': pred[:, 0],
            'disordered_score': pred[:, 1],
            'label': ref_hard,
        })
        partition_name = partition.replace('/', '_')
        pred_df.to_csv(os.path.join(output_path, f"predictions_{partition_name}.csv"), index=False)

    # Calculate metrics
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

