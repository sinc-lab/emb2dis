import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_disorder_prediction(centers, predictions, protein_id, threshold=0.5, 
                            output_path=None, figsize=(12, 5)):
    """
    Plot disorder predictions for a given protein.
    
    Args:
        centers: Array of center positions
        predictions: Prediction scores (structured, disordered) for each position
        protein_id: Protein identifier (acc)
        threshold: Threshold line used to classify residues as disordered
        output_path: Path to save the plot (None = display only)
        figsize: Figure size
    """
    pred = predictions.numpy()
    df = pd.DataFrame({
        'Position': centers,
        'Disorder Score': pred[:, 1]
    })

    sns.set_theme(style='ticks')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Identify disordered regions based on threshold
    disordered_mask = pred[:, 1] > threshold
    regions = []
    start = None
    for i, is_disordered in enumerate(disordered_mask):
        if is_disordered and start is None:
            start = i
        elif not is_disordered and start is not None:
            regions.append((centers[start], centers[i-1]))
            start = None
    if start is not None:
        regions.append((centers[start], centers[-1]))

    # Highlight disordered regions with background color
    for start, end in regions:
        ax.axvspan(start, end, alpha=0.15, color='red', zorder=0,
                    label='Disordered Region' if start == regions[0][0] else "")
    
    # Plot disorder score
    sns.lineplot(data=df, x='Position', y='Disorder Score', 
                 color="#d62728", linewidth=2, ax=ax, label='Disorder Score')
    
    # Plot threshold line
    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.5, 
               label=f'Threshold = {threshold}', zorder=1)
    
    # Set the rest of the plot aesthetics
    ax.set_xlim([centers.min(), centers.max()])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Prediction Score")
    ax.set_title(f"Disorder Prediction for {protein_id}", fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
   
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    ax.legend(by_label.values(), by_label.keys(), # loc='upper right', 
              frameon=True, facecolor='white', framealpha=0.8)
    # sns.despine()
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()