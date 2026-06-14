import os
import sys
import warnings
import torch as tr
sys.path.append(os.getcwd()) # to correctly import modules
from src.model import BaseModel
from src.utils import load_data
tr.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore", # Filter annoying warnings
                        message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")
warnings.simplefilter("ignore", FutureWarning)

def train(
        config: dict,
        output: str,
        ) -> float:
    """
    Trains a model based on the provided configuration and saves the best model weights.
    Args:
        config: Configuration dictionary containing model and training parameters.
        output: Directory where the model weights and training summary will be saved.
    Returns:
        The best AUC achieved during training.
    """

    categories = config['categories']
    device = config['device']
    num_workers = config['num_workers']

    weights_file = os.path.join(output, "weights.pk")
    summary = os.path.join(output, "train_summary.csv")

    # ------------------------------ DATALOADER ------------------------------ #
    train_csv = f"{config['data_path']}train.csv"
    print("Loading training data from", train_csv)
    train_loader, len_train = load_data(train_csv, config, is_segment=True, 
        is_training=True, categories=categories, num_workers=num_workers)
    
    # For early stopping, we use the dev set as a sliding window dataset
    dev_csv = f"{config['data_path']}dev.csv"
    print("Loading dev data from", dev_csv)
    dev_loader, len_dev = load_data(dev_csv, config, is_segment=False, 
        is_training=False, categories=categories, num_workers=num_workers)

    print("train", len_train, "dev", len_dev)
    
    # ------------------------------ LOAD MODEL ------------------------------ #
    net = BaseModel(len(categories), emb_size=config['emb_size'][config['pLM']], 
                    lr=config['lr'], device=device, p_dropout=config['p_dropout'],
                    filters=config['filters'], kernel_size=config['kernel_size'],
                    num_layers=config['n_resnet']) 

    # ------------------------------- TRAINING ------------------------------- #
    with open(summary, 'w') as s:
        s.write("Ep,Train loss,Dev Loss,Dev AUC,Dev error,Best AUC,Counter\n")
        INIT_EP, counter, best_auc = 0, 0, -1.0

    for epoch in range(INIT_EP, config['nepoch']):
        train_loss = net.fit(train_loader)
        dev_loss, dev_err, dev_auc, *_ = net.pred(dev_loader)

        # early stop
        sv_mod=""
        if dev_auc > best_auc:
            best_auc = dev_auc 
            tr.save(net.state_dict(), weights_file)
            counter = 0
            sv_mod=" - MODEL SAVED"
        else:
            counter += 1
            sv_mod=f" - EPOCH {counter} of {config['patience']}"

        print_msg=f"{epoch}: train loss {train_loss:.3f}, dev loss {dev_loss:.3f}, dev err {(dev_err*100):.2f}%, dev auc {(dev_auc*100):.2f}%"
        print(print_msg+sv_mod)

        with open(summary, 'a') as s:
            s.write(f"{epoch},{train_loss},{dev_loss},{dev_auc},{dev_err},{best_auc},{counter}\n")
        
        if counter >= config['patience']:
            return best_auc

# IMPROVE: add time taken for each epoch and resource usage