"""
SEED-VLA Dataset Preparation Script
Based on the NeuroLM framework

This script processes the SEED-VLA dataset which contains EEG data and PERCLOS indices
for classifying driver vigilance levels (awake, tired, drowsy).
"""

import mne
import numpy as np
import os
import pickle
import re
import glob
import argparse
import scipy.io as sio

# PERCLOS thresholds for classification
THRESHOLD_AWAKE_TIRED = 0.35
THRESHOLD_TIRED_DROWSY = 0.7

# Classification mapping
VIGILANCE_LABELS = {
    'awake': 0,  # PERCLOS < 0.35
    'tired': 1,  # 0.35 <= PERCLOS < 0.7
    'drowsy': 2  # PERCLOS >= 0.7
}

# Set standard channel order
chOrder_standard = ['EEG Pz']  # According to input description

def load_perclos_labels(perclos_file):
    """
    Load PERCLOS indices from MAT file
    Returns labels based on thresholds and their timestamps
    """
    try:
        # Load MATLAB file containing PERCLOS data
        mat_data = sio.loadmat(perclos_file)
        perclos = mat_data['perclos'].flatten()
        
        # Convert PERCLOS to vigilance labels using thresholds
        labels = np.zeros_like(perclos, dtype=int)
        labels[(perclos >= THRESHOLD_AWAKE_TIRED) & (perclos < THRESHOLD_TIRED_DROWSY)] = VIGILANCE_LABELS['tired']
        labels[perclos >= THRESHOLD_TIRED_DROWSY] = VIGILANCE_LABELS['drowsy']
        # perclos < THRESHOLD_AWAKE_TIRED remains 0 (awake)
        
        # Calculate timestamps based on 10 samples per minute
        # Convert to seconds (6 seconds per sample)
        timestamps = np.arange(len(perclos)) * 6.0
        
        return labels, timestamps
    except Exception as e:
        print(f"Error loading PERCLOS file {perclos_file}: {e}")
        raise

def find_perclos_file(eeg_file, perclos_dir):
    """
    Find the corresponding PERCLOS file for an EEG file
    Assumes the file naming convention is consistent between EEG and PERCLOS files
    """
    filename = os.path.basename(eeg_file)
    # Remove extension and any suffix to get the base identifier
    base_name = filename.split('.')[0]
    
    # Look for matching perclos file
    perclos_pattern = os.path.join(perclos_dir, f"{base_name}*.mat")
    perclos_files = glob.glob(perclos_pattern)
    
    if not perclos_files:
        raise FileNotFoundError(f"No PERCLOS file found for {eeg_file}. Searched pattern: {perclos_pattern}")
    
    # If multiple files found, use the first one
    return perclos_files[0]

def readEDF(eeg_file, perclos_file):
    """Read and preprocess EDF file from SEED-VLA dataset with corresponding PERCLOS labels"""
    try:
        print(f"Processing {eeg_file}")
        raw_data = mne.io.read_raw_edf(eeg_file, preload=True)
        
        # Extract EEG Pz channel if it exists
        if 'EEG Pz' not in raw_data.ch_names:
            available_channels = raw_data.ch_names
            print(f"Warning: 'EEG Pz' not found. Available channels: {available_channels}")
            # Try to find a similar channel (case-insensitive match for "Pz")
            pz_channels = [ch for ch in available_channels if 'pz' in ch.lower()]
            if pz_channels:
                raw_data.pick_channels([pz_channels[0]])
                print(f"Using channel {pz_channels[0]} instead")
            else:
                raise ValueError("No Pz channel found in the recording")
        else:
            raw_data.pick_channels(['EEG Pz'])
        
        # Get original sampling rate for resampling calculation
        orig_sfreq = raw_data.info['sfreq']
        print(f"Original sampling rate: {orig_sfreq} Hz")

        # check the frequency distribution of the eeg data
        raw_data.plot_psd(fmax=100)
        
        # Apply preprocessing
        print("Applying bandpass filter (0.1-75 Hz)")
        raw_data.filter(l_freq=0.1, h_freq=75.0)

        print("Applying 50 Hz notch filter")
        raw_data.notch_filter(50.0)
        
        print("Upsampling to 200 Hz")
        raw_data.resample(200, n_jobs=16)

        # check the frequency distribution of the eeg data
        raw_data.plot_psd(fmax=100)

        input("Press Enter to continue...")
        
        # Get EEG signal data in microvolts
        signals = raw_data.get_data(units='uV')
        times = raw_data.times
        
        # Load PERCLOS labels
        print(f"Loading PERCLOS labels from {perclos_file}")
        labels, label_times = load_perclos_labels(perclos_file)
        
        # Store the channel names before closing
        ch_names = raw_data.ch_names
        raw_data.close()
        
        return signals, times, labels, label_times, ch_names
    
    except Exception as e:
        print(f"Error processing {eeg_file}: {e}")
        raise

def BuildEvents(signals, times, labels, label_times, epoch_duration=30):
    """
    Build segments from continuous data based on PERCLOS labels
    Each segment corresponds to a PERCLOS measurement (every 6 seconds)
    """
    fs = 200.0  # New sampling rate is 200 Hz
    
    # Calculate samples per epoch
    epoch_len = int(fs * epoch_duration)
    features = []
    epoch_labels = []
    
    for i in range(len(label_times)):
        try:
            # Find the closest time point in the EEG data
            time_idx = np.argmin(np.abs(times - label_times[i]))
            
            # Check if we have enough signal after this point
            if time_idx + epoch_len > signals.shape[1]:
                print(f"Warning: Not enough signal data for epoch at {label_times[i]:.2f}s")
                continue
                
            # Extract epoch
            epoch = signals[:, time_idx:time_idx + epoch_len]
            features.append(epoch)
            epoch_labels.append(labels[i])
            
        except Exception as e:
            print(f"Warning: Error processing epoch at {label_times[i]:.2f}s - {e}")
            continue
    
    return np.array(features), np.array(epoch_labels).reshape(-1, 1)

def save_pickle(object, filename):
    """Save object as pickle file"""
    with open(filename, "wb") as f:
        pickle.dump(object, f)

def load_up_objects(eeg_files, perclos_dir, output_dir):
    """Process multiple EDF files with their PERCLOS labels and save as pickle objects"""
    for eeg_file in eeg_files:
        print(f"\nProcessing {eeg_file}")
        try:
            # Find corresponding PERCLOS file
            perclos_file = find_perclos_file(eeg_file, perclos_dir)
            print(f"Found PERCLOS file: {os.path.basename(perclos_file)}")
            
            # Process the data
            signals, times, labels, label_times, ch_names = readEDF(eeg_file, perclos_file)
            features, epoch_labels = BuildEvents(signals, times, labels, label_times)
            
            print(f"Created {len(features)} epochs for {eeg_file}")
            for idx, (signal, label) in enumerate(zip(features, epoch_labels)):                
                sample = {
                    "X": signal,
                    "ch_names": ch_names,
                    "y": int(label[0]),
                }
                
                # Create a clear, identifiable filename
                base_name = os.path.basename(eeg_file).replace('.edf', '')
                save_path = os.path.join(output_dir, f"{base_name}-epoch{idx}.pkl")
                
                # Save the sample
                save_pickle(sample, save_path)
            
        except Exception as e:
            print(f"Error processing {eeg_file}: {e}")
            continue

def process_dataset(eeg_dir, perclos_dir, output_dir):
    """Process the SEED-VLA dataset"""
    print(f"Processing SEED-VLA dataset from {eeg_dir}")
    
    # Create output directories
    train_out_dir = os.path.join(output_dir, "train")
    eval_out_dir = os.path.join(output_dir, "eval")
    test_out_dir = os.path.join(output_dir, "test")
    
    for directory in [train_out_dir, eval_out_dir, test_out_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Find all EEG files
    edf_files = []
    for dirName, _, fileList in os.walk(eeg_dir):
        for fname in fileList:
            if fname.endswith('.edf'):
                edf_files.append(os.path.join(dirName, fname))
    
    edf_files.sort()
    total_files = len(edf_files)
    
    if total_files == 0:
        raise ValueError(f"No EDF files found in {eeg_dir}")
    
    print(f"Found {total_files} EDF files")
    
    # Split into train/eval/test (60/20/20)
    train_size = int(0.6 * total_files)
    eval_size = int(0.2 * total_files)
    
    train_files = edf_files[:train_size]
    eval_files = edf_files[train_size:train_size+eval_size]
    test_files = edf_files[train_size+eval_size:]
    
    print(f"Split: {len(train_files)} train, {len(eval_files)} eval, {len(test_files)} test")
    
    # Process each split
    print("\nProcessing training files...")
    load_up_objects(train_files, perclos_dir, train_out_dir)
    
    print("\nProcessing evaluation files...")
    load_up_objects(eval_files, perclos_dir, eval_out_dir)
    
    print("\nProcessing test files...")
    load_up_objects(test_files, perclos_dir, test_out_dir)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    eeg_dir = r'H:\github\NeuroLM-main\datasets\VLA_VRW\lab\EEG'
    perclos_dir = r'H:\github\NeuroLM-main\datasets\VLA_VRW\lab\PERCLOS'
    output = r'H:\github\NeuroLM-main\out\VLA_VRW\lab'
    
    process_dataset(eeg_dir, perclos_dir, output)
