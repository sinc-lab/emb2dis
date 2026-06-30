import os
import re 
import torch
import tempfile
import esm
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
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


def get_esm2(sequences, protein_ids, output_dir, device='cuda'):
    device = _parse_device(device)
    
    # Load model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    num_repr_layer = 33  

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
                            return_contacts=False) # Contacts not needed
        
        for i, embed in enumerate(results["representations"][num_repr_layer]):
            # Extract embedding, remove special tokens (BOS and EOS)
            new_embed = embed.cpu().numpy()[1:batch_lens[0]-1]
            # Transpose to (emb_dim, L) format
            new_embed = new_embed.T
            np.save(os.path.join(output_dir, f'{prot_id}.npy'), arr=new_embed)
      

def generate_embeddings_from_fasta(
        fasta_path: str, 
        plm: str = 'ESM2', 
        verbose: bool = False, 
        device: str = 'cuda'
        ) -> list[tuple[torch.Tensor, str]]:
    """
    Generate embeddings from all sequences in a FASTA file on-the-fly.
    Args:
        fasta_path: Path to FASTA file
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
    
    if verbose:
        print(f"\nGenerating {plm} embeddings...")
        print(f"Using device: {device}")
    
    # Create temporary directory for embedding output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Generate embeddings using specified PLM
        if verbose:
            print(f"Loading {plm} model and generating embeddings...")
        
        if plm == 'ESM2':
            get_esm2(sequences=sequences, protein_ids=protein_ids, 
                     output_dir=str(temp_dir), device=device)
        else:
            raise ValueError(f"Unknown PLM: {plm}.")
        
        # Load the generated embeddings
        results = []
        for protein_id in protein_ids:
            emb_file = temp_dir / f"{protein_id}.npy"
            if not emb_file.exists():
                if verbose:
                    print(f"Warning: Failed to generate embedding for {protein_id}")
                continue
            
            emb = np.load(emb_file)
            results.append((torch.tensor(emb, dtype=torch.float32), protein_id))
        
        if verbose:
            print(f"Successfully generated {len(results)} embeddings.")
        
        return results