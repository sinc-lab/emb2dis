from torch.utils.data import Dataset
import torch as tr
import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path
import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def _soft_domain_score(window_start, window_end, domain_start, domain_end):
    """
    Calculate the fraction of a fixed-size window covered by a domain interval.
    Returns the proportion (0.0 to 1.0) of [window_start, window_end) that 
    overlaps with [domain_start, domain_end).
    """
    overlap_start = max(window_start, domain_start)
    overlap_end = min(window_end, domain_end)
    overlap_length = max(0, overlap_end - overlap_start)
    window_length = window_end - window_start
    
    if window_length == 0:
        return 0.0
    
    return overlap_length / window_length


@lru_cache(maxsize=32) # Reduced cache size to balance performance and memory usage
def _load_emb(emb_path_str, acc):
    file_path = os.path.join(emb_path_str, f"{acc}.npy") 
    emb = np.load(file_path) # (L, emb_dim)
    return tr.tensor(emb, dtype=tr.float32)


class SegmentDataset(Dataset): 
    """
    Dataset class for sampling regions of proteins with multiple domains
    or annotated regions. It samples ONE fixed-size window from each
    protein region for training or evaluation.
    """
    def __init__(self, dataset_path, emb_path, categories=("structured", "disordered"),
                 win_len=32, debug=False, is_training=False):
        """
        Dataset contains all valid segments in the complete proteins.
        """
        self.is_training = is_training
        self.emb_path = emb_path
        self.categories = categories
        self.win_len = win_len
        self.dataset = pd.read_csv(dataset_path)

        # If debugging, sample a smaller subset of the dataset
        if debug:
            self.dataset = self.dataset.sample(n=100)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """Sample one random or centered window from a region entry"""
        n_item = item
        item = self.dataset.iloc[item]

        emb = _load_emb(self.emb_path, item.acc)
        
        if item.start >= item.end: # Skip invalid domain and try the next sample
            return self.__getitem__((int(n_item) + 1) % len(self.dataset))

        # Determine the center of the window
        if self.is_training: # Randomly sample the center for training
            center = np.random.randint(item.start, item.end)
        else: # Use the midpoint for evaluation
            center = (item.start + item.end) // 2

        # Calculate the start and end of the window
        start = max(0, center - self.win_len//2)
        end = min(emb.shape[1], center + self.win_len//2)

        # Create a hard label based on the current item's label
        label_hard = self.categories.index(item.label)

        # Initialize the label vector. It's a soft label because it represents
        # the coverage of different classes in the window.
        label_soft = tr.zeros(len(self.categories))

        # Get all the domains/regions for the current protein
        domains = self.dataset[self.dataset.acc==item.acc]
        
        # Calculate coverage of the window on each domain to get a class score
        for k in range(len(domains)):
            score = _soft_domain_score(start, end, domains.iloc[k].start, domains.iloc[k].end)
            label_idx = self.categories.index(domains.iloc[k].label)
            label_soft[label_idx] += score

        # # force labels to sum 1 # * this was used in emb2pfam
        # s = label.sum()
        # if s<1: 
        #     ind = tr.where(label==0)[0]
        #     label[ind] = (1-s)/len(ind)

        # Create a fixed-size embedding window
        emb_win = tr.zeros((emb.shape[0], self.win_len), dtype=tr.float)
        emb_win[:,:end-start] = emb[:, start:end]

        return emb_win, label_soft, label_hard, center, item.acc, start, end
    

class AminoAcidDataset(Dataset):
    """
    This dataset samples a window for each annotated residue in a protein, to
    simulate a sliding window approach for evaluation.
    """
    def __init__(self, dataset_path, emb_path, categories=("structured", "disordered"),
                 win_len=32, step=1, debug=False):
    
        self.win_len = win_len
        self.half_win = win_len // 2
        self.emb_path = Path(emb_path)
        self.categories = categories
        # Map category (disorder state) to index 
        self.cat2idx = {c: i for i, c in enumerate(categories)}  

        # Load the dataset
        df = pd.read_csv(dataset_path).astype({"start": int, "end": int})

        if debug: # to debug, select only one acc randomly
            df = df.sample(n=1)
            
        # Get the domains/regions for each protein (acc)
        self.domains = {}
        for acc, g in df.groupby("acc"):
            self.domains[acc] = [# Store start, end, and label index
                (int(r.start), int(r.end), self.cat2idx[r.label])
                for r in g.itertuples(index=False)
            ]

        # Preload the lengths of protein embeddings
        prot_len = {
            acc: np.load(self.emb_path / f"{acc}.npy", mmap_mode="r").shape[1]
            for acc in self.domains.keys()
        }

        # Build a list of valid center positions
        examples = []
        for acc, doms in self.domains.items():
            L = prot_len[acc] 

            for start, end, label in doms:  # Iterate over domains
                for c in range(start, end + 1, step): # Include the end value in the range
                    start = max(0, c - self.half_win)
                    end = min(L, c + self.half_win)

                    if end - start > 0:  # Only include valid windows
                        examples.append((acc, c, label))

        if not examples:
            raise RuntimeError("No valid centres found: revise win_len or annotations")

        # Sort examples by protein accession and center position
        self.examples = sorted(examples, key=lambda t: (t[0], t[1]))

    def __len__(self):
        return len(self.examples)  

    def __getitem__(self, idx):
        """Sample one window centered on a residue."""
        acc, center, label_hard = self.examples[idx]  
        emb = _load_emb(self.emb_path, acc)
        L = emb.shape[1]

        # Calculate window boundaries
        win_start = max(0, center - self.half_win)
        win_end = min(L, center + self.half_win)

        # Compute soft scores for the window
        label_soft = np.zeros(len(self.categories), dtype=np.float32)
        for dom_start, dom_end, label in self.domains[acc]:  
                score = _soft_domain_score(win_start, win_end, dom_start, dom_end)
                label_soft[label] += score

        # Create a fixed-size embedding window
        win = np.zeros((emb.shape[0], self.win_len)) 
        win[:, :win_end-win_start] = emb[:, win_start:win_end]
        win = tr.tensor(win, dtype=tr.float32)

        return win, label_soft, label_hard, center, acc, win_start, win_end