"""
Predict disorder from protein embeddings using a trained model
"""
import argparse
import numpy as np
import torch as tr
import pandas as pd
from pathlib import Path
from src.model import BaseModel
from src.utils import ConfigLoader, predict_sliding_window, get_embedding_size, calculate_disorder_percentage
from src.plms import generate_embeddings_from_fasta
from src.plot import plot_disorder_prediction

def parser():
    parser = argparse.ArgumentParser(
        description='Predict disorder from protein embeddings using a trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        # to show the default values in help messages
    )
    parser.add_argument(
        '--fasta', '-f',
        type=str,
        required=True,
        help='Path to FASTA file (will generate embedding on-the-fly)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='ProtT5',
        choices=['ESM2', 'ProtT5'], # Later will add ['ProstT5', 'esmc_300m', 'esmc_600m'],
        help='Protein Language Model (pLM) used for generating embeddings. '
             'The disorder prediction model was trained using embeddings from this pLM'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results/',
        help='Output directory to save predictions (.csv) and plots (.png). '
             'If not provided, predictions and plots will be saved in the "results/" directory,'
             'with filenames based on the input FASTA file.'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda',
        help='Device to run predictions on (e.g., "cpu", "cuda", "cuda:0", "cuda:1")'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()

def main():
    args = parser()

    # Validate and setup device ------------------------------------------------
    device = args.device.lower()
    
    if device.startswith('cuda') and not tr.cuda.is_available():
        device = 'cpu'
        print("Warning: CUDA is not available. Switching to CPU.")
    
    if args.verbose:
        device_name = tr.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU'
        print(f"Using device: {device} ({device_name})")

    # Set up model path, config and weights ------------------------------------
    model_dir = Path(f"model/{args.model}/model0/")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    config_path = model_dir / 'config.yaml'
    weights_path = model_dir / 'weights.pk'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    # Load model configuration -------------------------------------------------
    if args.verbose:
        print(f"Model directory: {model_dir}")
        print(f"Loading configuration from: {config_path}")
    config_loader = ConfigLoader(model_path=str(config_path))
    config = config_loader.load()
    window_len = config.get('win_len', 13)
    threshold = config.get('threshold', 0.5)
    
    # Initialize model ---------------------------------------------------------
    if args.verbose:
        print(f"Loading model from: {weights_path}")
    categories = ['structured', 'disordered']
    model = BaseModel(
        len(categories),
        lr=config['lr'],
        device=device,
        emb_size=get_embedding_size(config.get('plm', 'ProtT5')),
        filters=config['filters'],
        kernel_size=config['kernel_size'],
        num_layers=config['n_resnet']
    )
    model.load_state_dict(tr.load(weights_path, map_location=device))
    model.eval()
    
    # Load FASTA and generate embeddings ---------------------------------------
    print(f"\nGenerating {args.model} embeddings for sequences in: {args.fasta}")
    results = generate_embeddings_from_fasta(
        fasta_path=args.fasta,
        plm=args.model, 
        verbose=args.verbose,
        device=device
    )
    
    # Predict disorder for all the proteins and save results -------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    # For each protein embedding and ID
    for emb, protein_id in results:
        if args.verbose:
            print(f"\n--- Processing Protein: {protein_id} ---")
            print(f"Sequence length: {emb.shape[1]} residues")
        
        # Predict --------------------------------------------------------------
        if args.verbose:
            print(f"Predicting disorder (window={window_len}) ")
        centers, predictions = predict_sliding_window(
            model, emb, window_len, step=1, 
            use_softmax=config.get('soft_max', True),
            median_filter_size=None  # No smoothing
        )
        
        # Calculate disorder percentage
        stats = calculate_disorder_percentage(predictions, 
                                              threshold=threshold)
        
        # Print results
        print(f"\nDISORDER PREDICTION RESULTS FOR: {protein_id}")
        print(f"Total residues:        {stats['total_residues']}")
        print(f"Disordered residues:   {stats['disordered_residues']}")
        print(f"Disorder percentage:   {stats['disorder_percentage']:.2f}%")
        
        # Save outputs ---------------------------------------------------------

        # Save plot
        output_plot = output_dir / f"{protein_id}_{args.model}_plot.png"
        plot_disorder_prediction(
            centers, 
            predictions, 
            protein_id, 
            threshold=threshold,
            output_path=output_plot
        )

        # Save predictions to CSV
        output_csv = output_dir / f"{protein_id}_{args.model}_predictions.csv"
        df = pd.DataFrame({
            'position': centers,
            'disordered_score': predictions[:, 1].numpy(),
            'predicted_label': (predictions[:, 1] > threshold).numpy().astype(int)
        })
        df.to_csv(output_csv, index=False)

        if args.verbose:
            print(f"Plot saved to: {output_plot}")
            print(f"Predictions saved to: {output_csv}")
        
        all_stats.append(stats)
    
    return all_stats

if __name__ == '__main__':
    main()