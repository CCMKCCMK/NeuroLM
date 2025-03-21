"""
ISRUC-Sleep Dataset Preparation Script
Based on the NeuroLM framework

This script processes the ISRUC-Sleep dataset which contains polysomnographic recordings.
"""

import numpy as np
import os
import pickle
import glob
import argparse
import mne
from scipy import signal
import shutil
import tempfile

# ISRUC-Sleep channel specifications
eog_channels = ['LOC-A2', 'ROC-A1', 'E1-M2', 'E2-M1']
eeg_channels = ['F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1',
                'F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1']
keep_channels = eog_channels + eeg_channels
written_channels = ['EOG E1', 'EOG E2', 'EEG F3', 'EEG C3', 'EEG O1', 'EEG F4', 'EEG C4', 'EEG O2']

_error_sets = []
# Sampling rate for ISRUC dataset
fs = 200.0  # Target sampling rate

def bandpass_filter(data, lowcut, highcut, fs):
    """Apply bandpass filter to the input data"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(5, [low, high], btype='band')
    
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, data[i])
    
    return filtered_data

def notch_filter(data, freq, fs, q=30.0):
    """Apply notch filter to remove power line noise"""
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = signal.iirnotch(w0, q)
    
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, data[i])
    
    return filtered_data

def save_pickle(object, filename):
    """Save object as pickle file"""
    with open(filename, "wb") as f:
        pickle.dump(object, f)

def read_label_file(label_path):
    """Read the label file containing sleep stage annotations"""
    with open(label_path, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    
    # Convert label 5 to 4 as required
    labels = [4 if l == 5 else l for l in labels]
    return np.array(labels)

def read_rec_file(rec_path, label_path):
    """Read .rec file (EDF format) and corresponding labels"""
    try:
        # Create a temporary copy of the .rec file with .edf extension
        # This is necessary because MNE only recognizes standard file extensions
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Copy the .rec file content to the temporary .edf file    
        shutil.copy2(rec_path, temp_path)
        
        try:
            # Read the temporary EDF file
            raw_data = mne.io.read_raw_edf(temp_path, preload=True)
        finally:
            # Delete the temporary file when done
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Check available channels
        available_channels = raw_data.ch_names
        print(f"Available channels: {available_channels}")
        
        # Pick only the channels we need
        channels_to_use = [ch for ch in keep_channels if ch in available_channels]
        if len(channels_to_use) != len(keep_channels):
            missing = set(keep_channels) - set(channels_to_use)
            print(f"Warning: Missing required channels: {missing}")
            print(f"Will proceed with available channels: {channels_to_use}")
        
        raw_data.pick_channels(channels_to_use)
        
        # Read labels
        labels = read_label_file(label_path)
        
        # Remove last 30 epochs to avoid noise
        num_epochs = len(labels) - 30
        if num_epochs <= 0:
            print(f"Warning: Not enough epochs in {rec_path}")
            return None, None, None
        
        labels = labels[:num_epochs]
        
        # Separate EOG and EEG channels
        eog_idx = [i for i, ch in enumerate(raw_data.ch_names) if ch in eog_channels]
        eeg_idx = [i for i, ch in enumerate(raw_data.ch_names) if ch in eeg_channels]
        
        # Resample to target frequency if needed
        if raw_data.info['sfreq'] != fs:
            raw_data.resample(fs)
        
        # Get data
        data = raw_data.get_data(units='uV')
        
        # Extract EOG and EEG data
        eog_data = data[eog_idx] if eog_idx else np.array([])
        eeg_data = data[eeg_idx] if eeg_idx else np.array([])
        
        # Filter EOG data (0.3-30 Hz)
        if eog_data.size > 0:
            eog_data = bandpass_filter(eog_data, 0.3, 30.0, fs)
        
        # Filter EEG data (0.1-75 Hz) and apply notch filter at 50Hz
        if eeg_data.size > 0:
            eeg_data = bandpass_filter(eeg_data, 0.1, 75.0, fs)
            eeg_data = notch_filter(eeg_data, 50.0, fs)
        
        # Combine filtered EOG and EEG data
        processed_data = np.vstack([eog_data, eeg_data]) if eog_data.size > 0 and eeg_data.size > 0 else (eog_data if eog_data.size > 0 else eeg_data)
        
        return processed_data, raw_data.ch_names, labels
    
    except Exception as e:
        print(f"Error processing {rec_path}: {e}")
        
        # Alternative approach if MNE fails: try using another library like pyedflib
        try:
            print("Trying alternative approach with pyedflib...")
            import pyedflib
            
            # Read the .rec file as binary data
            with pyedflib.EdfReader(rec_path) as f:
                n_channels = f.signals_in_file
                channel_names = f.getSignalLabels()
                
                # Map channel indices to the channels we want to keep
                indices_to_keep = []
                channels_kept = []
                
                for i, ch in enumerate(channel_names):
                    if ch in keep_channels:
                        indices_to_keep.append(i)
                        channels_kept.append(ch)
                
                if not indices_to_keep:
                    print(f"No required channels found in {rec_path}")
                    print(f"Available channels: {channel_names}")
                    return None, None, None
                
                # Read label file
                labels = read_label_file(label_path)
                
                # Remove last 30 epochs to avoid noise
                num_epochs = len(labels) - 30
                if num_epochs <= 0:
                    print(f"Warning: Not enough epochs in {rec_path}")
                    return None, None, None
                
                labels = labels[:num_epochs]
                
                # Read signal data
                data = np.zeros((len(indices_to_keep), f.getNSamples()[indices_to_keep[0]]))
                for i, ch_idx in enumerate(indices_to_keep):
                    data[i, :] = f.readSignal(ch_idx)
                
                # Get sampling frequency
                sample_rate = f.getSampleFrequency(indices_to_keep[0])
                
                # Separate EOG and EEG channels
                eog_idx = [i for i, ch in enumerate(channels_kept) if ch in eog_channels]
                eeg_idx = [i for i, ch in enumerate(channels_kept) if ch in eeg_channels]
                
                # Resample to target frequency if needed
                if sample_rate != fs:
                    print(f"Resampling from {sample_rate}Hz to {fs}Hz")
                    ratio = fs / sample_rate
                    new_length = int(data.shape[1] * ratio)
                    resampled_data = np.zeros((data.shape[0], new_length))
                    
                    for i in range(data.shape[0]):
                        resampled_data[i] = signal.resample(data[i], new_length)
                    
                    data = resampled_data
                
                # Extract EOG and EEG data
                eog_data = data[eog_idx] if eog_idx else np.array([])
                eeg_data = data[eeg_idx] if eeg_idx else np.array([])
                
                # Filter EOG data (0.3-30 Hz)
                if eog_data.size > 0:
                    eog_data = bandpass_filter(eog_data, 0.3, 30.0, fs)
                
                # Filter EEG data (0.1-75 Hz) and apply notch filter at 50Hz
                if eeg_data.size > 0:
                    eeg_data = bandpass_filter(eeg_data, 0.1, 75.0, fs)
                    eeg_data = notch_filter(eeg_data, 50.0, fs)
                
                # Combine filtered EOG and EEG data
                processed_data = np.vstack([eog_data, eeg_data]) if eog_data.size > 0 and eeg_data.size > 0 else (eog_data if eog_data.size > 0 else eeg_data)
                
                return processed_data, channels_kept, labels
            
        except ImportError:
            print("pyedflib not installed. Please install it with: pip install pyedflib")
            return None, None, None
        except Exception as e2:
            print(f"Alternative approach failed: {e2}")
            return None, None, None

def extract_epochs(data, labels, fs=200.0):
    """Extract 30-second epochs from continuous data"""
    epoch_samples = int(30 * fs)  # 30 seconds at sampling rate
    num_epochs = len(labels)
    
    # Create array to hold epoch data
    epochs = np.zeros((num_epochs, data.shape[0], epoch_samples))
    
    for i in range(num_epochs):
        start_sample = i * epoch_samples
        end_sample = (i + 1) * epoch_samples
        
        # Check if we have enough data for this epoch
        if end_sample <= data.shape[1]:
            epochs[i] = data[:, start_sample:end_sample]
        else:
            print(f"Warning: Not enough data for epoch {i}, padding with zeros")
            # Pad with zeros if we run out of data
            padding = end_sample - data.shape[1]
            epochs[i, :, :data.shape[1]-start_sample] = data[:, start_sample:]
            epochs[i, :, data.shape[1]-start_sample:] = 0
    
    return epochs

def process_subject(rec_path, label_path, output_dir, subject_id):
    """Process a single subject's recordings"""
    print(f"Processing subject {subject_id}: {rec_path}")
    
    # Read and preprocess data
    data, ch_names, labels = read_rec_file(rec_path, label_path)
    if data is None or labels is None or len(data) == 0:
        print(f"Error processing subject {subject_id}")
        _error_sets.append({
            "rec_path": rec_path,
            "label_path": label_path,
            'subject_id': subject_id,
            'ch_names': ch_names,
        })
        return 0
    
    # Extract epochs
    epochs = extract_epochs(data, labels, fs)
    print(f"Extracted {len(epochs)} epochs")
    
    # Save each epoch as a sample
    count = 0
    for idx, (epoch, label) in enumerate(zip(epochs, labels)):
        # Map actual channel names to standardized names
        # This ensures consistency regardless of the original file format
        sample = {
            "X": epoch,
            "ch_names": written_channels[:len(ch_names)],  # Use only as many channel names as we have channels
            "y": int(label),
        }
        
        # Save sample
        save_path = os.path.join(output_dir, f"subject{subject_id}-epoch{idx}.pkl")
        save_pickle(sample, save_path)
        count += 1
    
    return count

def process_dataset(input_dir, output_dir):
    """Process the entire ISRUC-Sleep dataset"""
    print(f"Processing ISRUC-Sleep dataset from {input_dir}")
    
    # Create output directories
    train_out_dir = os.path.join(output_dir, "train")
    eval_out_dir = os.path.join(output_dir, "eval")
    test_out_dir = os.path.join(output_dir, "test")
    
    for directory in [train_out_dir, eval_out_dir, test_out_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Find all subject folders
    subject_folders = []
    for item in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, item)
        if os.path.isdir(folder_path) and item.split('_')[0].isdigit():
            subject_folders.append(folder_path)
    
    subject_folders.sort()
    total_subjects = len(subject_folders)
    
    if total_subjects == 0:
        raise ValueError(f"No subject folders found in {input_dir}")
    
    print(f"Found {total_subjects} subjects")
    
    # Split into train/eval/test (60/20/20)
    train_size = int(0.6 * total_subjects)
    eval_size = int(0.2 * total_subjects)
    
    train_folders = subject_folders[:train_size]
    eval_folders = subject_folders[train_size:train_size+eval_size]
    test_folders = subject_folders[train_size+eval_size:]
    
    print(f"Split: {len(train_folders)} train, {len(eval_folders)} eval, {len(test_folders)} test")
    
    # Process each split
    total_samples = 0
    
    print("\nProcessing training subjects...")
    for folder in train_folders:
        subject_id = os.path.basename(folder).split('_')[0]
        rec_file = os.path.join(folder, f"{subject_id.split('_')[-1]}.rec")
        label_file = os.path.join(folder, "label.txt")
        
        if not os.path.exists(rec_file) or not os.path.exists(label_file):
            print(f"Warning: Missing files for subject {subject_id}")
            continue
        
        samples = process_subject(rec_file, label_file, train_out_dir, subject_id)
        total_samples += samples
    
    print("\nProcessing evaluation subjects...")
    for folder in eval_folders:
        subject_id = os.path.basename(folder).split('_')[0]
        rec_file = os.path.join(folder, f"{subject_id.split('_')[-1]}.rec")
        label_file = os.path.join(folder, "label.txt")
        
        if not os.path.exists(rec_file) or not os.path.exists(label_file):
            print(f"Warning: Missing files for subject {subject_id}")
            continue
        
        samples = process_subject(rec_file, label_file, eval_out_dir, subject_id)
        total_samples += samples
    
    print("\nProcessing test subjects...")
    for folder in test_folders:
        subject_id = os.path.basename(folder).split('_')[0]
        rec_file = os.path.join(folder, f"{subject_id.split('_')[-1]}.rec")
        label_file = os.path.join(folder, "label.txt")
        
        if not os.path.exists(rec_file) or not os.path.exists(label_file):
            print(f"Warning: Missing files for subject {subject_id}")
            continue
        
        samples = process_subject(rec_file, label_file, test_out_dir, subject_id)
        total_samples += samples
    
    print(f"\nProcessing complete! Total samples created: {total_samples}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ISRUC-Sleep dataset for NeuroLM')
    
    parser.add_argument('--input', type=str, default=r"E:\dataset\ISRUC-SLEEP\subgroupIII",
                        help='Input directory containing the ISRUC-Sleep dataset')
                        
    parser.add_argument('--output', type=str, default=r"E:\dataset\ISRUC",
                        help='Output directory for processed data')
    
    # Add an argument to specify which file format handling to prioritize
    parser.add_argument('--file_handling', type=str, default='auto',
                        choices=['auto', 'copy_as_edf', 'pyedflib'],
                        help='Approach to handle .rec files: auto (try both), copy_as_edf, or pyedflib')
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)
