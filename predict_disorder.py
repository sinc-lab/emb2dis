"""
Predict disorder from protein embeddings using a trained model
"""
import argparse
import yaml
import time
import torch as tr
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.ensemble import EnsembleModel, build_ensemble_dirs
from src.utils import get_embedding_size, calculate_disorder_percentage

THRESHOLD = 0.5

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
        default='ESM2',
        choices=['ESM2'], # Later will add ['ProstT5', 'esmc_300m', 'esmc_600m'],
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
    parser.add_argument(
        '--embeddings-dir', '-e',
        type=str,
        default=None,
        help='Directory with pre-computed embeddings, one {protein_id}.npy per FASTA record. '
    )
    parser.add_argument(
        '--caid',
        action='store_true',
        help='Emit CAID-format outputs: one {id}.caid per protein and a single timings.csv '
             'in --output-dir. When set, the legacy CSV and PNG plot are not written.'
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
    ensemble_root = Path(f"model/ensemble_{args.model}")
    if not ensemble_root.exists():
        raise FileNotFoundError(f"Ensemble model directory not found: {ensemble_root}")

    # Load model configuration -------------------------------------------------
    if args.verbose:
        print(f"Ensemble directory: {ensemble_root}")
        
    with open('config/env.yaml', 'r') as f:
        config = yaml.safe_load(f)

    threshold = THRESHOLD
    emb_size = get_embedding_size(config.get('pLM', args.model))

    # Initialize ensemble ------------------------------------------------------
    model_dirs = build_ensemble_dirs(config)
    if args.verbose:
        print(f"Loading ensemble members from {len(model_dirs)} fold directories")
    model = EnsembleModel(model_dirs, device=device)

    # Load FASTA and obtain embeddings -----------------------------------------
    if args.embeddings_dir:
        print(f"\nLoading pre-computed {args.model} embeddings from: {args.embeddings_dir}")
        from src.caid_io import load_embeddings_from_dir
        emb_results = load_embeddings_from_dir(
            fasta_path=args.fasta,
            embeddings_dir=args.embeddings_dir,
            emb_size=emb_size,
            verbose=args.verbose,
        )
    else:
        print(f"\nGenerating {args.model} embeddings for sequences in: {args.fasta}")
        from src.plms import generate_embeddings_from_fasta
        emb_results = [
            (emb, pid, None)
            for emb, pid in generate_embeddings_from_fasta(
                fasta_path=args.fasta,
                plm=args.model,
                verbose=args.verbose,
                device=device,
            )
        ]

    # Predict disorder for all the proteins and save results -------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    timings = []  # list of (protein_id, milliseconds)
    total_proteins = len(emb_results)

    # For each protein embedding and ID
    with tqdm(total=total_proteins, unit='protein', desc='Analyzing proteins', dynamic_ncols=True) as pbar:
        for emb, protein_id, sequence in emb_results:
            pbar.set_description(f"Analyzing {protein_id}")

            if args.verbose:
                print(f"\n--- Processing Protein: {protein_id} ---")
                print(f"Sequence length: {emb.shape[1]} residues")
        
            # Predict --------------------------------------------------------------
            t0 = time.perf_counter()
            centers, predictions = model.pred_sliding_window(emb, step=1)
            predictions = tr.as_tensor(predictions, dtype=tr.float)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append((protein_id, elapsed_ms))

            # Calculate disorder percentage
            stats = calculate_disorder_percentage(predictions,
                                                  threshold=threshold)

            # Print results (verbose mode only)
            if args.verbose:
                print(f"\nDISORDER PREDICTION RESULTS FOR: {protein_id}")
                print(f"Total residues:        {stats['total_residues']}")
                print(f"Disordered residues:   {stats['disordered_residues']}")
                print(f"Disorder percentage:   {stats['disorder_percentage']:.2f}%")

            # Save outputs ---------------------------------------------------------

            if args.caid:
                # CAID-format output only
                from src.caid_io import write_caid_file
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
                    threshold=threshold,
                )
                if args.verbose:
                    print(f"CAID file saved to: {caid_path}")
            else:
                # Local enriched outputs (plot + CSV) -------------------------------------
                from src.plot import plot_disorder_prediction
                output_plot = output_dir / f"{protein_id}_{args.model}_plot.png"
                plot_disorder_prediction(
                    centers,
                    predictions,
                    protein_id,
                    threshold=threshold,
                    output_path=output_plot,
                )
                if args.verbose:
                    print(f"Plot saved to: {output_plot}")

                output_csv = output_dir / f"{protein_id}_{args.model}_predictions.csv"
                df = pd.DataFrame({
                    'position': centers+1,
                    'disordered_score': predictions[:, 1].numpy(),
                    'predicted_label': (predictions[:, 1] > threshold).numpy().astype(int)
                })
                df.to_csv(output_csv, index=False)
                if args.verbose:
                    print(f"Predictions saved to: {output_csv}")

            all_stats.append(stats)
            pbar.update(1)

    if args.caid:
        from src.caid_io import write_timings_csv
        timings_path = output_dir / "timings.csv"
        write_timings_csv(timings_path, f"emb2dis-{args.model}", timings)
        if args.verbose:
            print(f"Timings saved to: {timings_path}")

    return all_stats

if __name__ == '__main__':
    main()