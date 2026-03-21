import os
import re 
import torch
import tempfile
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
      
def compute_esmc_embed(sequence, model="esmc_300m", device="cuda"):
    # TODO: add this model
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

def get_esmc(sequences, protein_ids, output_dir, esmc_model, device='cuda'):
    # TODO: add this model
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

    # Load the tokenizer and encoder model for ProtT5
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', 
                                            do_lower_case=False,
                                            legacy=True)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

    # Use full precision on CPU and half precision on GPU to save memory
    model.full() if device == 'cpu' else model.half()

    model = model.eval()

    for prot_id, seq in tqdm(zip(protein_ids, sequences)):
        seq_len = len(seq)

        # Only process sequences shorter than 4000 residues
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
            new_embed = numpy_embedding[0][:seq_len, :].T  # Transpose to [embedding_dim, sequence_length]

            # Save the embedding as a .npy file using the sequence ID
            np.save(os.path.join(output_dir, f'{prot_id}.npy'), arr=new_embed)
        else:
            print(f"Warning: Sequence {prot_id} is too long ({seq_len} residues) for ProtT5. Skipping.")


def get_ProstT5(sequences, protein_ids, output_dir, device='cuda'):
    device = _parse_device(device)

    # Load the ProstT5 tokenizer and encoder model from Hugging Face
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', 
                                            do_lower_case=False, 
                                            legacy=False)
    model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

    # Use full precision on CPU and half precision on GPU
    model.full() if device == 'cpu' else model.half()
    model = model.eval()

    for prot_id, seq in tqdm(zip(protein_ids, sequences)):
        seq_len = len(seq)

        # Insert spaces between residues
        seq_processed = " ".join(list(seq))

        # Add ProstT5-specific prompt token at the start of each sequence
        seq_processed = "<AA2fold> " + seq_processed

        # Tokenize sequences using the ProstT5 tokenizer
        ids = tokenizer.batch_encode_plus([seq_processed], add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Disable gradient tracking for inference
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)

        numpy_embedding = embedding.last_hidden_state.cpu().numpy()

        # Remove the first token (corresponding to the <AA2fold> prompt)
        # Keep only the embeddings for actual amino acid residues
        new_embed = numpy_embedding[0][1:seq_len+1, :].T  # Transpose to [embedding_dim, sequence_length]

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
        seq = re.sub(r"[UZOB]", "X", seq.upper())
        
        # Truncate length > 4000
        if len(seq) > 4000:
            print(f"Warning: Sequence {r.id} is too long ({len(seq)} residues) for ProtT5/ProstT5. "
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
        elif plm in ['esmc_300m', 'esmc_600m']:
            get_esmc(sequences=sequences, protein_ids=protein_ids, 
                     output_dir=str(temp_dir), esmc_model=plm, device=device)
        elif plm == 'ProtT5':
            get_ProtT5(sequences=sequences, protein_ids=protein_ids, 
                       output_dir=str(temp_dir), device=device)
        elif plm == 'ProstT5':
            get_ProstT5(sequences=sequences, protein_ids=protein_ids, 
                        output_dir=str(temp_dir), device=device)
        else:
            raise ValueError(f"Unknown PLM: {plm}. Choose from: ESM2, ProtT5") # Later will add ProstT5, esmc_300m, esmc_600m
        
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