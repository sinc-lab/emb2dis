import csv
import time
import numpy as np
import torch as tr
from pathlib import Path
from Bio import SeqIO


def load_embeddings_from_dir(fasta_path: str, embeddings_dir: str,
                             emb_size: int, verbose: bool = False) -> list:
    """
    Load pre-computed embeddings for each FASTA record from a directory. Expected format is (emb_dim, L).

    Args:
        fasta_path: Path to FASTA file.
        embeddings_dir: Directory containing one {id}.npy per FASTA record.
        emb_size: Expected embedding dimension for the pLM (used to disambiguate
            the (emb_dim, L) vs (L, emb_dim) orientation of the saved arrays).
        verbose: Print progress information.

    Returns:
        list of (embedding_tensor, protein_id, sequence) tuples,
        where embedding_tensor has shape (emb_dim, L) and sequence is the
        original FASTA sequence.
    """
    embeddings_dir = Path(embeddings_dir)
    if not embeddings_dir.is_dir():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    with open(fasta_path, "r") as f:
        records = list(SeqIO.parse(f, "fasta"))

    if not records:
        raise ValueError(f"No sequences found in FASTA file: {fasta_path}")

    results = []
    for record in records:
        protein_id = record.id
        sequence = str(record.seq).upper()

        emb_path = embeddings_dir / f"{protein_id}.npy"
        if not emb_path.exists():
            print(f"Warning: embedding not found for {protein_id} at {emb_path}. Skipping.")
            continue

        emb = np.load(emb_path)

        if verbose:
            print(f"Loaded embedding for {protein_id}: shape={emb.shape}")

        results.append((tr.tensor(emb, dtype=tr.float32), protein_id, sequence))

    if verbose:
        print(f"Loaded {len(results)} embeddings from {embeddings_dir}")
    return results


def write_caid_file(output_path, protein_id: str, sequence: str,
                    centers, predictions, threshold: float = 0.5):
    """
    Write a per-protein CAID-format prediction file.

    Format:
        >{protein_id}
        {pos}\\t{residue}\\t{score:.3f}\\t{label}
        ...

    Args:
        output_path: Path to write the .caid file.
        protein_id: Protein identifier (FASTA record id).
        sequence: Cleaned amino-acid sequence (one letter per residue).
        centers: 1D array of residue positions (0-indexed in the embedding).
        predictions: Tensor of shape (L, 2) with [structured, disordered] probs.
        threshold: Disorder probability threshold for the binary label.
    """
    scores = predictions[:, 1].numpy()
    labels = (scores > threshold).astype(int)

    with open(output_path, "w") as f:
        f.write(f">{protein_id}\n")
        for pos_idx, score, label in zip(centers, scores, labels):
            residue = sequence[pos_idx] if pos_idx < len(sequence) else "X"
            f.write(f"{pos_idx + 1}\t{residue}\t{score:.3f}\t{int(label)}\n")


def write_timings_csv(output_path, predictor_name: str, rows: list):
    """
    Write the CAID timings.csv file.

    Format:
        # Running {predictor_name}, started {ctime}
        sequence,milliseconds
        {protein_id},{ms}
        ...

    Args:
        output_path: Path to write timings.csv.
        predictor_name: Name displayed in the header comment.
        rows: List of (protein_id, milliseconds) tuples.
    """
    started = time.strftime("%a %b %e %H:%M:%S %Z %Y")
    with open(output_path, "w", newline="") as f:
        f.write(f"# Running {predictor_name}, started {started}\n")
        writer = csv.writer(f)
        writer.writerow(["sequence", "milliseconds"])
        for protein_id, ms in rows:
            writer.writerow([protein_id, int(round(ms))])
