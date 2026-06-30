"""
Predict disorder from protein embeddings using a trained model

Usage: 
    python predict_disorder.py --fasta <path_to_fasta> --embedding-dir <path_to_embeddings_dir> --output-dir <output_dir> [--device <device>] [--verbose] [--threads <num_threads>]

For example:
    python predict_disorder.py --fasta data/samples.fasta --embedding-dir data/embeddings/
"""
import argparse
import yaml
import time
import torch as tr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.ensemble import EnsembleModel, build_ensemble_dirs
from src.utils import get_embedding_size
from src.caid_io import write_timings_csv, write_caid_file

THRESHOLD = 0.5
MODEL = 'ESM2'

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
        '--embedding-dir', '-e',
        type=str,
        required=True,
        help='Directory with pre-computed embeddings, one {protein_id}.npy per FASTA record. '
    )    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results/',
        help='Output directory to save predictions'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cpu',
        help='Device to run predictions on (e.g., "cpu", "cuda", "cuda:0", "cuda:1")'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=None,
        help='Cap the number of CPU threads torch uses (torch.set_num_threads).'
    )
    return parser.parse_args()

def main():
    args = parser()

    # Cap CPU threads if requested ---------------------------------------------
    if args.threads is not None:
        tr.set_num_threads(args.threads)
        try:
            tr.set_num_interop_threads(1)
        except RuntimeError:
            # set_num_interop_threads must be called before any parallel work;
            # if torch already initialized the pool, silently ignore.
            pass

    # Validate and setup device ------------------------------------------------
    device = args.device.lower()
    
    if device.startswith('cuda') and not tr.cuda.is_available():
        device = 'cpu'
        print("Warning: CUDA is not available. Switching to CPU.")
    
    if args.verbose:
        device_name = tr.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU'
        print(f"Using device: {device} ({device_name})")

    # Set up ensemble path and configuration ----------------------------------
    with open('config/env.yaml', 'r') as f:
        config = yaml.safe_load(f)

    main_model_dir = config.get('main_model_dir')
    if not main_model_dir:
        raise ValueError("Ensemble model directory not specified in config.")

    emb_size = get_embedding_size(config.get('pLM', 'ESM2'))

    # Initialize ensemble ------------------------------------------------------
    model_dirs = build_ensemble_dirs(config)
    if args.verbose:
        print(f"Loading ensemble members from {len(model_dirs)} fold directories")
    model = EnsembleModel(model_dirs, device=device)

    # Load FASTA and obtain embeddings -----------------------------------------
    if args.embedding_dir:
        print(f"\nLoading pre-computed ESM-2 embeddings from: {args.embedding_dir}")
        from src.caid_io import load_embeddings_from_dir
        emb_results = load_embeddings_from_dir(
            fasta_path=args.fasta,
            embeddings_dir=args.embedding_dir,
            emb_size=emb_size,
            verbose=args.verbose,
        )
    else:
        print(f"\nGenerating ESM-2 embeddings for sequences in: {args.fasta}")
        from src.plms import generate_embeddings_from_fasta
        emb_results = [
            (emb, pid, None)
            for emb, pid in generate_embeddings_from_fasta(
                fasta_path=args.fasta,
                plm='ESM2',
                verbose=args.verbose,
                device=device,
            )
        ]

    # Predict disorder for all the proteins and save results -------------------
    output_dir = Path(args.output_dir) / 'disorder'
    output_dir.mkdir(parents=True, exist_ok=True)

    timings = []  # list of (protein_id, milliseconds)
    total_proteins = len(emb_results)

    # For each protein embedding and ID
    print(f"\nPredicting disorder for {total_proteins} proteins...")
    with tqdm(total=total_proteins, unit='protein', desc='Analyzing proteins', dynamic_ncols=True) as pbar:
        for emb, protein_id, sequence in emb_results:
            pbar.set_description(f"Analyzing {protein_id}")
        
            # Predict --------------------------------------------------------------
            t0 = time.perf_counter()
            centers, predictions = model.pred_sliding_window(emb, step=1)
            predictions = tr.as_tensor(predictions, dtype=tr.float)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append((protein_id, elapsed_ms))

            # Save outputs ---------------------------------------------------------
            if sequence is None:
                from Bio import SeqIO
                with open(args.fasta, 'r') as fh:
                    rec = next((r for r in SeqIO.parse(fh, 'fasta') if r.id == protein_id), None)
                sequence = str(rec.seq).upper() if rec is not None else 'X' * len(predictions)
            caid_path = output_dir / f"{protein_id}.caid"
            write_caid_file(
                caid_path,
                protein_id,
                sequence,
                centers,
                predictions,
                threshold=THRESHOLD,
            )
            pbar.update(1)

    print(f"\nPrediction files saved to: {output_dir}/")
        
    timings_path = output_dir / "timings.csv"
    write_timings_csv(timings_path, "e_emb2dis", timings)
    print(f"Timings saved to: {timings_path}\n")


if __name__ == '__main__':
    main()