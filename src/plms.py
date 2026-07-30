"""
To generate embeddings:
    python src/plms.py <fasta_path> <output_dir> --plm <PLM_NAME> --verbose --device <DEVICE>
For example:
    python src/plms.py data/samples.fasta data/embeddings --plm ESM2_35m --verbose --device cuda:1
"""

import os
import re 
import torch
import argparse
import esm
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from pathlib import Path


def _parse_device(device: str) -> torch.device:
    """
    Parse device string ('cpu', 'cuda', 'cuda:0', etc.) and return 
    torch.device object.
    """
    if isinstance(device, str):
        if device == 'cuda':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    raise ValueError(f"Invalid device type: {type(device)}. Expected str")


def get_esm2(
        sequences: list[str],
        protein_ids: list[str],
        output_dir: str,        
        model_name: str = "esm2",
        device='cuda'
        ):
    """Compute ESM2 embeddings for a list of protein sequences."""
    # Available ESM2 models and their representation layers and embedding dimensions (E)
    MODELS = { # Model name: (model_id, representation_layer, embedding_dim, model_size)
        "esm2_650m": ("esm2_t33_650M_UR50D", 33, 1280, "650M"),
        "esm2": ("esm2_t33_650M_UR50D", 33, 1280, "650M"),
        "esm2_150m": ("esm2_t30_150M_UR50D", 30, 640, "150M"),
        "esm2_35m": ("esm2_t12_35M_UR50D", 12, 480, "35M"),
        "esm2_8m": ("esm2_t6_8M_UR50D", 6, 320, "8M"),
    }
    device = _parse_device(device)
    
    # Load model
    try:
        model_id, num_repr_layer, _, _ = MODELS[model_name]
        pretrained_plm = getattr(esm.pretrained, model_id)
        model, alphabet = pretrained_plm()
    except AttributeError:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    for prot_id, seq in tqdm(zip(protein_ids, sequences)):
        data = [(prot_id, seq)]          

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[num_repr_layer], 
                            return_contacts=False)
        
        for i, embed in enumerate(results["representations"][num_repr_layer]):
            new_embed = embed.cpu().numpy()[1:batch_lens[0]-1] # Extract and remove special tokens
            new_embed = new_embed.T # Transpose to (emb_dim, L)
            np.save(os.path.join(output_dir, f'{prot_id}.npy'), arr=new_embed)

      
def compute_esmc_embed(sequence, model="esmc_300m", device="cuda"): # TODO: add this model
    from esm.models.esmc import ESMC # move to top if implemented!
    from esm.sdk.api import ESMProtein, LogitsConfig

    protein = ESMProtein(sequence=sequence)

    client = ESMC.from_pretrained(model).to(device)

    protein_tensor = client.encode(protein)

    # Run the model to obtain per-residue embeddings
    logits_output = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True)
    )

    return logits_output.embeddings


def get_esmc(sequences, protein_ids, output_dir, esmc_model, device='cuda'): # TODO: add this model
    device = _parse_device(device)

    for prot_id, seq in tqdm(zip(protein_ids, sequences)):
        embedding = compute_esmc_embed(
            sequence=seq,
            model=esmc_model,
            device=str(device)
        ).cpu().numpy()[0]

        batch_lens = embedding.shape[0]

        # Remove the first ([CLS]) and last ([EOS]) token embeddings
        embedding = embedding[1:batch_lens - 1]

        np.save(os.path.join(output_dir, f'{prot_id}.npy'), arr=embedding)


def get_ProtT5(sequences, protein_ids, output_dir, device='cuda'):
    device = _parse_device(device)

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', 
                                            do_lower_case=False,
                                            legacy=True)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

    model.full() if device == 'cpu' else model.half()
    model = model.eval()

    for prot_id, seq in tqdm(zip(protein_ids, sequences)):
        seq_len = len(seq)

        if seq_len < 4000:
            # Insert spaces between residues (as required by ProtT5)
            seq_processed = " ".join(list(seq))

            # Tokenize sequences and pad to the longest sequence
            ids = tokenizer.batch_encode_plus([seq_processed], add_special_tokens=True, padding="longest")

            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            with torch.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask)

            numpy_embedding = embedding.last_hidden_state.cpu().numpy()

            # Remove padding: keep only the first `seq_len` embeddings
            new_embed = numpy_embedding[0][:seq_len, :].T  # Transpose to (emb_dim, L)

            np.save(os.path.join(output_dir, f'{prot_id}.npy'), arr=new_embed)
        else:
            print(f"Warning: Sequence {prot_id} is too long ({seq_len} residues) for ProtT5. Skipping.")


def get_ProstT5(sequences, protein_ids, output_dir, device='cuda'):
    device = _parse_device(device)

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', 
                                            do_lower_case=False, 
                                            legacy=False)
    model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

    model.full() if device == 'cpu' else model.half()
    model = model.eval()

    for prot_id, seq in tqdm(zip(protein_ids, sequences)):
        seq_len = len(seq)

        # Insert spaces between residues (as required by ProtT5)
        seq_processed = " ".join(list(seq))

        # Add ProstT5-specific prompt token at the start of each sequence
        seq_processed = "<AA2fold> " + seq_processed

        ids = tokenizer.batch_encode_plus([seq_processed], add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Disable gradient tracking for inference
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)

        numpy_embedding = embedding.last_hidden_state.cpu().numpy()

        # Remove the first token (corresponding to the <AA2fold> prompt)
        # Keep only the embeddings for actual amino acid residues
        new_embed = numpy_embedding[0][1:seq_len+1, :].T  # Transpose to (emb_dim, L)

        np.save(os.path.join(output_dir, f'{prot_id}.npy'), arr=new_embed)


def generate_embeddings_from_fasta(
        fasta_path: str, 
        output_dir: str,
        plm: str = 'ESM2',
        verbose: bool = False, 
        device: str = 'cuda'
        ) -> list[tuple[torch.Tensor, str]]:
    """
    Generate embeddings from all sequences in a FASTA file on-the-fly.
    Args:
        fasta_path: Path to FASTA file
        output_dir: Directory to save the generated embeddings
        plm: Protein language model to use ('ESM2', 'ProtT5')
        verbose: Print progress information
        device: Device to use ('cpu', 'cuda', 'cuda:0', etc.)
    Returns:
        list: List of tuples (embedding tensor of shape (emb_dim, L), protein_id)
    """
    if verbose:
        print(f"Reading FASTA file: {fasta_path}")
        
    with open(fasta_path, 'r') as f:
        records = list(SeqIO.parse(f, "fasta"))
    if not records:
        raise ValueError(f"No sequences found in FASTA file: {fasta_path}")
    
    if verbose:
        print(f"Found {len(records)} sequences in FASTA file.")
    
    protein_ids = [r.id for r in records]
    sequences = []
    for r in records:
        seq = str(r.seq)
        # Clean sequence (replace unusual amino acids with X)
        seq = re.sub(r"[UZOBJ]", "X", seq.upper())
        
        # Truncate length > 4000
        if len(seq) > 4000:
            print(f"Warning: Sequence {r.id} is too long ({len(seq)} residues). "
                  f"Truncating to 4000 residues.")
            seq = seq[:4000]
        sequences.append(seq)
    
    # Create output directory for embedding output
    plm_output_dir = os.path.join(output_dir, plm)
    os.makedirs(plm_output_dir, exist_ok=True)
    plm_output_dir_str = str(plm_output_dir)

    # Generate embeddings using specified PLM
    if verbose:
        print(f"Loading {plm} model and generating embeddings...")
        print(f"Using device: {device}")
    
    args = {
        'sequences': sequences,
        'protein_ids': protein_ids,
        'output_dir': plm_output_dir_str,
        'device': device
    }

    if plm.lower().startswith('esm2'):
        get_esm2(**args, model_name=plm.lower())
    elif plm.lower() == 'prott5':
        get_ProtT5(**args)
    elif plm.lower() == 'prostt5':
        get_ProstT5(**args)
    elif plm.lower().startswith('esmc'):
        raise NotImplementedError("ESMC model support is not yet implemented.")
    else:
        raise ValueError(f"Unsupported PLM: {plm}")
    
    # Load the generated embeddings
    for protein_id in protein_ids:
        emb_file = Path(plm_output_dir) / f"{protein_id}.npy"
        if not emb_file.exists():
            print(f"Warning: Failed to generate embedding for {protein_id}")

    # TODO: add logger!

def parser():
    parser = argparse.ArgumentParser(description="Generate embeddings from a FASTA file using a specified protein language model.")
    parser.add_argument("fasta_path", type=str, help="Path to the input FASTA file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the generated embeddings.")
    parser.add_argument("--plm", type=str, default="ESM2", # ESM2 is ESM2_650M, it's written like this for backward compatibility
                        choices=["ESM2", "ESM2_8m", "ESM2_35m", "ESM2_150m", "ESM2_650m", "ProtT5", "ProstT5"], 
                        help="Protein language model to use for embedding generation.")
    parser.add_argument("--verbose", action="store_true", help="Print progress information.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation (e.g., 'cpu', 'cuda', 'cuda:0').")
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    generate_embeddings_from_fasta(
        fasta_path=args.fasta_path,
        output_dir=args.output_dir,
        plm=args.plm,
        verbose=args.verbose,
        device=args.device
    )
