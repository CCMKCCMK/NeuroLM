"""
Sleep-EDF Dataset Preparation Script
Based on the NeuroLM framework - Improved version

This script processes the Sleep-EDF Dataset Expanded from PhysioNet
which contains whole-night polysomnographic sleep recordings.
"""

import mne
import numpy as np
import os
import pickle
import re
import glob
import argparse

# Sleep-EDF uses these channels
drop_channels = ['EMG submental', 'Event marker', 'Temp rectal', 'Resp oro-nasal', ]
# For both SC (Sleep Cassette) and ST (Sleep Telemetry) files
chOrder_standard = ['EOG horizontal', 'EEG Fpz-Cz', 'EEG Pz-Oz']

fs = 200.0

symbols_sleep_edf = {
    'Sleep stage W': 0,   # Wake
    'Sleep stage 1': 1,   # Stage 1
    'Sleep stage 2': 2,   # Stage 2
    'Sleep stage 3': 3,   # Stage 3
    'Sleep stage 4': 3,   # Stage 4 (usually combined with Stage 3 in modern scoring)
    'Sleep stage R': 4,   # REM
    'Movement time': 5,   # Movement time
    'Sleep stage ?': 6    # Unknown
}

def parse_hypnogram(hypno_file):
    """Parse the hypnogram EDF+ file"""
    try:
        annotations = mne.read_annotations(hypno_file)
        labels = []
        onsets = []
        
        for annot in annotations:
            if annot['description'] in symbols_sleep_edf:
                labels.append(symbols_sleep_edf[annot['description']])
                onsets.append(annot['onset'])
            else:
                print(f"Warning: Unknown sleep stage '{annot['description']}' at {annot['onset']}")
                raise ValueError(f"Unknown sleep stage '{annot['description']}'")
        
        return np.array(labels), np.array(onsets) * 100 
    except Exception as e:
        print(f"Error parsing hypnogram {hypno_file}: {e}")
        raise

def BuildEvents(signals, times, labels, onsets):
    """Build 30-second epoch segments from continuous data based on hypnogram markers"""
    
    # 30-second epochs for sleep staging as per R&K manual
    epoch_len = int(fs * 30)
    features = []
    epoch_labels = []
    
    for i in range(len(onsets)-1):
        if labels[i] > 4:
            continue
        try:
            start_indices = np.where(times >= onsets[i])[0]
            end_indices = np.where(times >= onsets[i+1])[0]
            
            if len(start_indices) == 0 or len(end_indices) == 0:
                print(f"Warning: No matching time points for epoch {i} (onset {onsets[i]:.2f}s to {onsets[i+1]:.2f}s)")
                continue
                
            start = start_indices[0]
            end = end_indices[0]

            # print real time of the epoch
            # print(f"Epoch {i}: {onsets[i]:.2f}s - {onsets[i+1]:.2f}s")
            
            for j in range(start, end, epoch_len):
                if j + epoch_len > end:
                    break
                
                epoch = signals[:, j:j+epoch_len]
                features.append(epoch)
                epoch_labels.append(labels[i])

        except IndexError:
            print(f"Warning: Could not process epoch {i} - time index out of range")
            continue
    
    return np.array(features), np.array(epoch_labels).reshape(-1, 1)

def find_hypno_file(psg_path):
    """
    Find the corresponding hypnogram file for a PSG file
    
    The Sleep-EDF dataset naming convention:
    - PSG files: SC4ssNEO-PSG.edf (SC=Sleep Cassette) or ST7ssNJ0-PSG.edf (ST=Sleep Telemetry)
    - Hypnogram files: SC4ssNEX-Hypnogram.edf or ST7ssNJ0-Hypnogram.edf
    
    where:
    - ss is the subject number
    - N is the night
    - E/J is a recording identifier
    - O is typically 0
    - X is the scorer ID (the 8th letter of the filename)
    """
    filename = os.path.basename(psg_path)
    dir_path = os.path.dirname(psg_path)
    
    # Extract the first 7 characters which identify the recording
    # e.g., 'SC4001E' from 'SC4001E0-PSG.edf'
    base_prefix = filename[:7]
    
    # Search for hypnogram files matching this pattern
    hypno_pattern = os.path.join(dir_path, f"{base_prefix}*-Hypnogram.edf")
    hypno_files = glob.glob(hypno_pattern)
    
    if not hypno_files:
        raise FileNotFoundError(f"No Hypnogram file found for {psg_path}. Searched pattern: {hypno_pattern}")
    
    # If multiple files found, use the first one (can be refined based on specific needs)
    return hypno_files[0]

def readEDF(fileName):
    """Read and preprocess an EDF file from the Sleep-EDF dataset"""
    try:
        print(f"Processing {fileName}")
        Rawdata = mne.io.read_raw_edf(fileName, preload=True)
        
        if drop_channels is not None:
            # Only drop channels that exist in the recording
            useless_chs = [ch for ch in drop_channels if ch in Rawdata.ch_names]
            if useless_chs:
                Rawdata.drop_channels(useless_chs)
        
        # Extract available channels from standard order list
        available_channels = [ch for ch in chOrder_standard if ch in Rawdata.ch_names]
        
        if not available_channels:
            raise ValueError(f"No standard channels found in {fileName}. Available: {Rawdata.ch_names}")
        
        # Pick only the channels we want
        Rawdata.pick_channels(available_channels)

        # Find corresponding hypnogram file
        try:
            hypno_file = find_hypno_file(fileName)
            print(f"Found hypnogram file: {os.path.basename(hypno_file)}")
            labels, onsets = parse_hypnogram(hypno_file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        
        # Filter out non-sleep stages (0 is Wake, 1-4 are sleep stages, 5 is Movement, 6 is Unknown)
        sleep_indices = np.where((labels >= 1) & (labels <= 4))[0]
        
        if len(sleep_indices) == 0:
            print(f"Warning: No sleep periods found in {fileName}")
            # Process the entire recording if no sleep periods found
            _, times = Rawdata[:]
            signals = Rawdata.get_data(units='uV')
        else:
            print(f"Trimming data to include only 30 minutes before first sleep and 30 minutes after last sleep")
            print(f"Pre-sleep start: {onsets[sleep_indices[0]]:.2f}s, Post-sleep end: {onsets[sleep_indices[-1]]:.2f}s")
            
            # Find first and last sleep periods
            first_sleep_idx = sleep_indices[0]
            last_sleep_idx = sleep_indices[-1]
            
            # Get onset times for these periods
            first_sleep_onset = onsets[first_sleep_idx]
            last_sleep_onset = onsets[last_sleep_idx]
            
            # Add 30 seconds to last sleep onset to include the full epoch
            last_sleep_end = last_sleep_onset + 30
            
            # Calculate 30 minutes before and after in seconds
            thirty_min = 30 * 60 * 100
            start_time = max(0, first_sleep_onset - thirty_min)
            end_time = last_sleep_end + thirty_min
            

            print(f"Start time: {start_time:.2f}s, End time: {end_time:.2f}s")
            
            # Get timing information
            start_sample = int(start_time/100*fs)
            end_sample = int(end_time/100*fs)
            
            time_Rawdata = Rawdata.copy().resample(fs, n_jobs=16)
            
            # Extract only the relevant portion of the data
            _, times = time_Rawdata[:, start_sample:end_sample]
            signals = np.zeros((len(Rawdata.ch_names), len(times)))
            
            # Adjust onsets to match the new time reference            
            onsets = onsets - start_time
            # Remove any annotations that fall outside our trimmed data
            valid_indices = np.where(onsets < (end_time - start_time))[0]
            onsets[0] = 0
            onsets = onsets[valid_indices]
            labels = labels[valid_indices]
            times = times * fs
            times = times - times[0]
            onsets = onsets*(fs/100)

        
        # Create copies of the raw data for separate EOG and EEG processing
        eog_channels = [ch for ch in Rawdata.ch_names if 'EOG' in ch]
        eeg_channels = [ch for ch in Rawdata.ch_names if 'EEG' in ch]
        
        print(f"EOG channels: {eog_channels}")
        print(f"EEG channels: {eeg_channels}")
        eog_raw = Rawdata.copy().pick_channels(eog_channels)
        eeg_raw = Rawdata.copy().pick_channels(eeg_channels)

        # Process EOG channels (0.3-30 Hz)
        # Process EEG channels (0.1-75 Hz with 50Hz notch) NO NEED HERE (SAMPLED AT 100Hz)
        eog_raw.filter(l_freq=0.3, h_freq=30.0, n_jobs=16)

        eog_raw.resample(fs, n_jobs=16)
        eeg_raw.resample(fs, n_jobs=16)

        # Get processed data back into a combined dataset
        for i, ch in enumerate(Rawdata.ch_names):
            if ch in eog_channels:
                idx = eog_raw.ch_names.index(ch)
                signals[i, :] = eog_raw.get_data(units='uV')[idx, start_sample:end_sample]
            elif ch in eeg_channels:
                idx = eeg_raw.ch_names.index(ch)
                signals[i, :] = eeg_raw.get_data(units='uV')[idx, start_sample:end_sample]

        # Store the original channel names before closing
        orig_ch_names = Rawdata.ch_names
        Rawdata.close()
        
        if eog_channels:
            eog_raw.close()
        if eeg_channels:
            eeg_raw.close()
            
        return [signals, times, labels, onsets, orig_ch_names]
    
    except Exception as e:
        print(f"Error processing {fileName}: {e}")
        raise

def load_up_objects(fileList, OutDir):
    """Process multiple EDF files and save as pickle objects"""
    for fname in fileList:
        print(f"\nProcessing {fname}")
        try:
            [signals, times, labels, onsets, ch_names] = readEDF(fname)
            features, epoch_labels = BuildEvents(signals, times, labels, onsets)
            
            print(f"Created {len(features)} epochs for {fname}")
            for idx, (signal, label) in enumerate(zip(features, epoch_labels)):                
                sample = {
                    "X": signal,
                    "ch_names": ch_names,
                    "y": int(label[0]),
                }
                
                # Create a clear, identifiable filename
                base_name = os.path.basename(fname).replace('-PSG.edf', '')
                save_path = os.path.join(OutDir, f"{base_name}-epoch{idx}.pkl")
                
                # Save the sample
                save_pickle(sample, save_path)
                
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

def save_pickle(object, filename):
    """Save object as pickle file"""
    with open(filename, "wb") as f:
        pickle.dump(object, f)

def process_dataset(input_dir, output_dir):
    """Process the entire Sleep-EDF dataset"""
    print(f"Processing Sleep-EDF dataset from {input_dir}")
    
    # Create output directories
    train_out_dir = os.path.join(output_dir, "train")
    eval_out_dir = os.path.join(output_dir, "eval")
    test_out_dir = os.path.join(output_dir, "test")
    
    for directory in [train_out_dir, eval_out_dir, test_out_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Find all PSG files (from both studies)
    edf_files = []
    for dirName, _, fileList in os.walk(input_dir):
        for fname in fileList:
            if fname.endswith('-PSG.edf'):
                edf_files.append(os.path.join(dirName, fname))
    
    edf_files.sort()
    total_files = len(edf_files)
    
    if total_files == 0:
        raise ValueError(f"No PSG files found in {input_dir}")
    
    print(f"Found {total_files} PSG files")
    
    # Split into train/eval/test (60/20/20)
    # For reproducibility, sort files before splitting
    train_size = int(0.6 * total_files)
    eval_size = int(0.2 * total_files)
    
    train_files = edf_files[:train_size]
    eval_files = edf_files[train_size:train_size+eval_size]
    test_files = edf_files[train_size+eval_size:]
    
    print(f"Split: {len(train_files)} train, {len(eval_files)} eval, {len(test_files)} test")
    
    # Process each split
    print("\nProcessing training files...")
    load_up_objects(train_files, train_out_dir)
    
    print("\nProcessing evaluation files...")
    load_up_objects(eval_files, eval_out_dir)
    
    print("\nProcessing test files...")
    load_up_objects(test_files, test_out_dir)
    
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Sleep-EDF dataset for NeuroLM')
    # ['sleep-cassette', 'sleep-telemetry']:

    parser.add_argument('--input', type=str, default=r"D:\datasets\physionet.org\files\sleep-edfx\1.0.0\sleep-telemetry",
                        help='Input directory containing the Sleep-EDF dataset')
                        
    parser.add_argument('--output', type=str, default=r"E:\dataset\SleepEDF_new\sleep-telemetry",
                        help='Output directory for processed data')
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)
