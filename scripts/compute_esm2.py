"""
Read a FASTA, generate ESM-2 embeddings on CPU (full precision), and save one {protein_id}.npy per record to an output dir. This script uses the full dev environment (requirements.txt). 
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plms import get_esm2
from Bio import SeqIO
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fasta', '-f', required=True, help='Input FASTA file')
    p.add_argument('--output-dir', '-o', required=True,
                   help='Directory to write {protein_id}.npy files')
    p.add_argument('--device', '-d', default='cpu',
                   help='Device for ESM-2 (default: cpu; matches container precision)')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.fasta, 'r') as f:
        records = list(SeqIO.parse(f, 'fasta'))

    protein_ids, sequences = [], []
    for r in records:
        seq = re.sub(r'[UZOBJ]', 'X', str(r.seq).upper())
        if len(seq) > 4000:
            print(f"Warning: truncating {r.id} from {len(seq)} to 4000 residues")
            seq = seq[:4000]
        protein_ids.append(r.id)
        sequences.append(seq)

    get_esm2(sequences=sequences, protein_ids=protein_ids,
               output_dir=str(out_dir), device=args.device)
    print(f"Wrote {len(protein_ids)} embeddings to {out_dir}")


if __name__ == '__main__':
    main()
