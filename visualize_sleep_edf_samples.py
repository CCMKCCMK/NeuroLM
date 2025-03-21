"""
Sleep-EDF Dataset Visualization Script
For visualizing processed sample files from the Sleep-EDF dataset
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Sleep stage labels for display
sleep_stage_labels = {
    0: 'Wake',
    1: 'Stage 1',
    2: 'Stage 2', 
    3: 'Stage 3/4',
    4: 'REM',
    5: 'Movement',
    6: 'Unknown'
}

def load_pickle(filepath):
    """Load a pickle file and return its contents"""
    try:
        with open(filepath, 'rb') as f:
            sample = pickle.load(f)
        return sample
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def visualize_sample(sample, title=None, save_path=None):
    """Visualize a sleep-EDF sample with its label"""
    if sample is None:
        print("No sample provided")
        return
    
    # Extract data
    X = sample['X']  # Shape: [n_channels, n_samples]
    ch_names = sample['ch_names']
    label = sample['y']
    
    # Convert label to Python int if it's a numpy array or other type
    if isinstance(label, (np.ndarray, np.generic)):
        label = int(label.item())
    elif not isinstance(label, int):
        label = int(label)
    
    # Get number of channels
    n_channels = X.shape[0]
    n_samples = X.shape[1]
    
    # Create time vector (30 seconds at fs=200Hz)
    time = np.arange(n_samples) / 200.0
    
    # Create figure
    fig = plt.figure(figsize=(15, 8))
    
    # Set up GridSpec for main plot and label
    gs = GridSpec(n_channels+1, 1, height_ratios=[3]*n_channels + [1])
    
    # Plot each channel
    for i in range(n_channels):
        ax = fig.add_subplot(gs[i])
        ax.plot(time, X[i], linewidth=0.5)
        ax.set_ylabel(f"{ch_names[i]}\n(μV)")
        
        # Set y-limits based on signal range with some padding
        signal_max = np.max(np.abs(X[i]))
        y_limit = max(100, signal_max * 1.2)  # At least ±100 μV or 120% of max amplitude
        ax.set_ylim(-y_limit, y_limit)
        
        # Only show x-axis for the bottom plot
        if i < n_channels - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)")
    
    # Create a panel for the label at the bottom
    ax_label = fig.add_subplot(gs[-1])
    ax_label.axis('off')
    
    # Display the sleep stage label prominently
    stage_text = sleep_stage_labels.get(label, f"Unknown ({label})")
    ax_label.text(0.5, 0.5, f"Sleep Stage: {stage_text}", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=16,
                 bbox=dict(facecolor='lightblue', alpha=0.5))
    
    # Set title if provided
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize Sleep-EDF processed samples')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the pickle file to visualize')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save the visualization (optional)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.file):
        print(f"Error: File {args.file} does not exist")
        return
    
    # Load and visualize sample
    sample = load_pickle(args.file)
    if sample:
        filename = os.path.basename(args.file)
        visualize_sample(sample, title=f"Sleep-EDF Sample: {filename}", save_path=args.save)

if __name__ == "__main__":
    main()
