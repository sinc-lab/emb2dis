"""
Generate protein embeddings from a FASTA file using a specified protein
language model (pLM).

Usage:
    python -m scripts.generate_embeddings <fasta_path> <output_dir> --plm <PLM_NAME> --verbose --device <DEVICE>

Arguments:
    fasta_path      Path to the input FASTA file.
    output_dir      Directory to save the generated embeddings.
    --plm           Protein language model to use for embedding generation.
                     One of: ESM2, ESM2_8m, ESM2_35m, ESM2_150m, ESM2_650m,
                     ProtT5, ProstT5. Defaults to ESM2.
    --verbose       Print progress information.
    --device        Device to use for computation (e.g. 'cpu', 'cuda',
                     'cuda:0'). Defaults to 'cuda'.

For example:
    python -m scripts.generate_embeddings data/samples.fasta embeddings --plm ESM2 --verbose --device cuda
"""
import os
import sys
sys.path.append(os.getcwd())

from src.plms import generate_embeddings_from_fasta, parser


def main():
    args = parser().parse_args()
    generate_embeddings_from_fasta(
        fasta_path=args.fasta_path,
        output_dir=args.output_dir,
        plm=args.plm,
        verbose=args.verbose,
        device=args.device
    )


if __name__ == "__main__":
    main()
