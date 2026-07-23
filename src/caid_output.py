import pandas as pd
from pathlib import Path


def format_caid_rows(centers, sequence, scores, labels):
    """Build (pos, aa, score, label) tuples for one protein."""
    rows = []
    for idx, score, label in zip(centers, scores, labels):
        idx = int(idx)
        aa = sequence[idx]         # centers are 0-based
        pos = idx + 1              # 1-based position for output
        rows.append((pos, aa, float(score), int(label)))
    return rows


def write_caid_block(out, acc, rows):
    out.write(f">{acc}\n")
    for pos, aa, score, label in rows:
        out.write(f"{pos}\t{aa}\t{score:.3f}\t{label}\n")


def load_fasta_sequences(fasta_path: Path) -> dict:
    """Load accession -> sequence mappings from a FASTA file."""
    sequences = {}
    current_acc = None
    current_sequence = []

    with open(fasta_path, 'r') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if current_acc is not None:
                    sequences[current_acc] = ''.join(current_sequence)
                current_acc = line[1:].split()[0]
                current_sequence = []
            else:
                current_sequence.append(line)

    if current_acc is not None:
        sequences[current_acc] = ''.join(current_sequence)

    return sequences


def save_partition_predictions_caid(output_dir: Path, partition: str, names: list,
                                    centers: list, scores: list, labels: list,
                                    config: dict) -> Path:
    """Save per-residue predictions in CAID format for one partition."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    partition_name = partition.replace('/', '_')
    caid_path = output_dir / f"predictions_{partition_name}.caid"
    partition_key = partition.lower().split('/')[-1]
    if partition_key not in {'disorder_pdb', 'disorder_nox'}:
        raise ValueError(
            f"Available data: disorder_pdb or disorder_nox, got {partition!r}"
        )

    fasta_path = Path(config['caid_path']).parent / 'raw' / f'{partition_key}.fasta'
    sequences = load_fasta_sequences(fasta_path)

    with open(caid_path, 'w') as out:
        current_acc = None
        current_rows = []

        for acc, center, score, label in zip(names, centers, scores, labels):
            sequence = sequences[acc]
            row = (int(center) + 1, sequence[int(center)], float(score), int(label))

            if acc != current_acc:
                if current_rows:
                    write_caid_block(out, current_acc, current_rows)
                current_acc = acc
                current_rows = []

            current_rows.append(row)

        if current_rows:
            write_caid_block(out, current_acc, current_rows)

    return caid_path


def save_protein_prediction_caid(acc: str, rows: list, output_dir: Path) -> Path:
    """Save predictions for a single protein to a CAID format file."""
    predictions_caid = output_dir / f"{acc}.caid"
    with open(predictions_caid, 'w') as out:
        write_caid_block(out, acc, rows)
    return predictions_caid


def save_combined_predictions(all_rows: list, dir: Path, save_csv: bool = False,
                              file_name: str = "all_predictions"): # not used!
    """Write combined CSV and CAID files from in-memory prediction results."""
    if not all_rows:
        return None, None
    combined_caid = dir / f"{file_name}.caid"

    if save_csv:
        combined_csv = dir / f"{file_name}.csv"
        csv_rows = []

    with open(combined_caid, 'w') as out:
        for result in all_rows:
            write_caid_block(out, result['protein_id'], result['rows'])
            if save_csv:
                csv_rows.extend(
                    {'protein_id': result['protein_id'], 
                     'position': pos, 'aa': aa, 
                     'score': score, 'label': label}
                    for pos, aa, score, label in result['rows']
                )
    if save_csv:
        pd.DataFrame(csv_rows).to_csv(combined_csv, index=False)
    return combined_caid


def save_prediction_timings(timings: list, dir: Path, initial_time: str, 
                            model_name: str = "e_emb2bind") -> Path:
    """Save per-sequence prediction timings to a CSV file."""
    if not timings:
        return None
    
    time_str = f"{initial_time.strftime('%a %b %e %H:%M:%S %Z %Y')}"

    timings_csv = dir / "timings.csv"

    with open(timings_csv, 'w') as f:
        f.write(f"# Running {model_name}, started {time_str}\n")
        f.write("sequence,milliseconds\n")
        for seq_id, ms in timings:
            f.write(f"{seq_id},{ms}\n")

    return timings_csv