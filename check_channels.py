"""
Utility script to check what channel names are available in the standard_1020 list
"""

from dataset import PickleLoader, standard_1020
import argparse
import os
import pickle
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Check channel names in dataset')
    parser.add_argument('--dataset_dir', type=str, default='/d:/datasets/NeuroLM/processed/SleepEDF',
                        help='Path to the dataset directory')
    args = parser.parse_args()
    
    # Print all available channel names in the standard_1020 list
    print("Available channel names in standard_1020:")
    for i, ch in enumerate(standard_1020):
        print(f"  {i}: '{ch}'")
    
    # Check a few samples from the dataset
    print("\nChecking samples from the dataset...")
    train_dir = os.path.join(args.dataset_dir, 'train')
    files = list(Path(train_dir).glob('*.pkl'))
    
    if files:
        # Load a few samples directly to examine them
        print(f"\nFound {len(files)} files in {train_dir}")
        print("Examining first few samples:")
        
        for i, file_path in enumerate(files[::len(files)//10]):
            try:
                with open(file_path, 'rb') as f:
                    sample = pickle.load(f)
                
                print(f"  Sample {i}:")
                print(f"    File: {file_path}")
                print(f"    Channel names: {sample['ch_names']}")
                print(f"    Label: {sample['y']}")
                
                # Check if channel names are in standard_1020
                chans = []
                for ch_name in sample['ch_names']:
                    try:
                        idx = standard_1020.index(ch_name)
                        chans.append(idx)
                    except ValueError:
                        print(f"    WARNING: Channel '{ch_name}' not found in standard_1020")
                
                if chans:
                    print(f"    Channel indices: {chans}")
                else:
                    print("    No valid channels found in standard_1020 list")
                    
            except Exception as e:
                print(f"    Error processing {file_path}: {e}")
    else:
        print(f"No sample files found in {train_dir}")

if __name__ == "__main__":
    main()
