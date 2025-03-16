"""
Locked-In State Dataset Preparation Script
Based on the NeuroLM framework and adapted from vhdr2mat.m

This script processes EEG and EOG data from locked-in state patients
who used an auditory EOG-based communication system.
"""

import os
import numpy as np
import pickle
import mne
import re
from glob import glob
import argparse

# Session types defined in the original dataset
SESSION_TYPES = {
    "Training": 0,
    "Feedback": 1, 
    "Speller": 2  # Renamed from "Copy_spelling"/"Free_spelling" to match the actual folder name
}

# Define event type mappings 
EVENT_TYPE_MAPPING = {
    # 9: 'SESSION_START',  # S9 - Start of session

    1: 'FEEDBACK_YES',   # S1 - Feedback for Yes
    2: 'FEEDBACK_NO',    # S2 - Feedback for No
    3: 'FEEDBACK_SPELL', # S3 - Other feedback

    5: 'QUESTION_YES',   # S5 - Yes question
    6: 'QUESTION_NO',    # S6 - No question
    7: 'SPELLING_OPTION',# S7 - Spelling option

    4: 'RESPONSE_YES',       # S4 - Response segment
    8: 'RESPONSE_NO',       # S8 - Response segment (alternative)
    13: 'RESPONSE_SPELL',      # S13 - Response segment (alternative)

    10: 'BASELINE_YES',  # S10 - Baseline for Yes
    11: 'BASELINE_NO',   # S11 - Baseline for No
    12: 'BASELINE_SPELL',   # S12 - Baseline for spelling

    15: 'SESSION_END',   # S15 - End of session
}

def process_vhdr_file(vhdr_path):
    """Process a single .vhdr file and extract relevant data"""
    print(f"Processing {vhdr_path}...")
    
    # Define standard EOG channels that should be present in all outputs
    STANDARD_EOG_CHANNELS = [
        'EOGR', 'EOGDR', 'EOGRU', 'EOGRD', 'EOGUR', 'EOGDiagR', 
        'EOGUL', 'EOGDiagLU', 'EOGU', 'EOGD', 'EOGL', 'EOGDL', 'EOGLH', 'EOGRH'
    ]
    
    try:
        # Load the .vhdr file using MNE
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
        
        # Extract events/annotations BEFORE preprocessing (like HMC)
        events_data = mne.events_from_annotations(raw)
        event_array = events_data[0]  # This is a numpy array of shape (n_events, 3)
        event_id_to_desc = events_data[1]  # This is a dict mapping event IDs to descriptions
        
        # Extract onset markers correctly
        onset = np.zeros(len(raw.times), dtype=int)
        event_types = []
        event_samples = []
        event_values = []
        
        # Correctly iterate through the event array
        for i in range(event_array.shape[0]):
            event_sample = event_array[i, 0]  # Sample index
            event_id = event_array[i, 2]      # Event ID
            event_desc = list(event_id_to_desc.keys())[list(event_id_to_desc.values()).index(event_id)]
            
            # Process event descriptions to extract trigger values
            if event_desc.startswith('S'):
                try:
                    # Extract numeric part of the S-marker (e.g., S1, S2, etc.)
                    event_val = int(event_desc.split()[-1]) if len(event_desc) <= 3 else int(event_desc.split()[-1])
                    
                    # Only keep events with meaningful labels
                    if event_val in EVENT_TYPE_MAPPING:
                        onset[event_sample] = event_val
                        event_samples.append(event_sample)
                        event_values.append(event_val)
                        event_types.append(EVENT_TYPE_MAPPING[event_val])
                except (ValueError, IndexError):
                    # Skip events that can't be parsed properly
                    continue
        
        # Skip if no meaningful events found
        if not event_samples:
            print(f"No meaningful events found in {vhdr_path}, skipping")
            return None
            
        # Rest of the function remains the same...
        # Filter for EOG channels only
        eog_channels = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
        if not eog_channels:
            print(f"No EOG channels found in {vhdr_path}, skipping")
            return None
        
        # Pick only EOG channels
        raw.pick_channels(eog_channels)
        sfreq = raw.info['sfreq']

        # Apply preprocessing steps (same order as HMC)
        print("Applying filters")
        try:
            raw.filter(l_freq=0.1, h_freq=35.0, picks='all')
            raw.notch_filter(50.0, picks='all')
        except Exception as e:
            print(f"Warning: Filtering error - {e}")
            print("Continuing without filtering...")
        
        print("Resampling to 100 Hz")
        raw.resample(100, n_jobs=20)

        # scale the event labels to match the new sampling rate
        event_samples = [int(sample * 100 / sfreq) for sample in event_samples]
        
        # Get file name without extension
        eegname = os.path.basename(vhdr_path)[:-5]
        
        # Extract directory structure information
        folder_path = os.path.dirname(vhdr_path)
        session_type = os.path.basename(folder_path)
        
        # Get the raw data
        data = raw.get_data()
        available_channels = raw.ch_names
        
        # Create a new data matrix with all standard channels
        n_samples = data.shape[1]
        standard_data = np.zeros((len(STANDARD_EOG_CHANNELS), n_samples))
        
        # Map existing data to standard channel locations
        for i, ch in enumerate(STANDARD_EOG_CHANNELS):
            if ch in available_channels:
                ch_idx = available_channels.index(ch)
                standard_data[i] = data[ch_idx]
        
        # Create a data structure similar to the MATLAB code
        eeg_data = {
            "rawFileName": eegname,
            "SessionType": session_type,
            "Channels": STANDARD_EOG_CHANNELS,
            "TriggerSequence": event_types,  # Use meaningful event types
            "Data": standard_data,
            "EventsOnset": onset,
            "EventSamples": event_samples,  # Store the actual sample indices
            "EventValues": event_values,    # Store the actual event values
            "TimeVector": raw.times / 1000.0
        }
        
        return eeg_data
        
    except Exception as e:
        print(f"Error processing {vhdr_path}: {e}")
        return None

def extract_features_and_labels(eeg_data, segment_length=3):
    """Extract features and labels in the same format as HMC dataset"""
    fs = 200  # Sampling frequency
    signals = eeg_data["Data"]
    event_samples = eeg_data["EventSamples"]  # Use the stored sample indices
    event_values = eeg_data["EventValues"]    # Use the stored event values
    num_channels = signals.shape[0]
    
    # Ensure there are enough events to create segments
    if len(event_samples) < 2:
        return np.array([]), np.array([]), {}
    
    # Count valid segments (only count those with known labels)
    valid_segments = 0
    for i in range(len(event_samples)-1):
        start_idx = event_samples[i]
        end_idx = event_samples[i+1]
        
        # Check if this segment has a meaningful label
        trigger_value = event_values[i]
        if trigger_value in EVENT_TYPE_MAPPING:
            segment_length_samples = end_idx - start_idx
            num_complete_segments = segment_length_samples // (fs * segment_length)
            valid_segments += num_complete_segments # At least one segment
    
    if valid_segments == 0:
        return np.array([]), np.array([]), {}
    
    # Pre-allocate arrays
    features = np.zeros([valid_segments, num_channels, int(fs) * segment_length])
    labels = np.zeros([valid_segments, 1])

    
    # Extract features and labels
    segment_idx = 0
    for i in range(len(event_samples)-1):
        start_idx = event_samples[i]
        end_idx = event_samples[i+1]
        
        # Get trigger value
        trigger_value = event_values[i]
        if trigger_value not in EVENT_TYPE_MAPPING:
            continue

        # Calculate segment duration in samples
        segment_duration = end_idx - start_idx
        num_complete_segments = segment_duration // (fs * segment_length)

        for j in range(num_complete_segments):
            # Calculate segment start index
            seg_start_idx = event_samples[i] + j * fs * segment_length
            seg_end_idx = event_samples[i] + (j + 1) * fs * segment_length

            # Extract segment data
            segment_data = signals[:, seg_start_idx:seg_end_idx]

            # Skip if too short to exactly match segment_length
            if segment_data.shape[1] < fs * segment_length:
                continue  # Skip this segment as it's too short

            # Store in features and labels arrays
            features[segment_idx] = segment_data
            # labels[segment_idx] = label_map[trigger_value]  # Use mapped label
            labels[segment_idx] = trigger_value  # Use mapped label
            segment_idx += 1
    
    # Trim any unused slots
    if segment_idx < valid_segments:
        features = features[:segment_idx]
        labels = labels[:segment_idx]
    
    return features, labels

def process_patient_data(patient_dir, out_dir):
    """
    Process all data for a single patient and save in NeuroLM format
    
    This follows the hierarchical structure in the original dataset:
    Patient -> Visit -> Day -> Session Type -> Recording
    """
    print(f"Processing patient directory: {patient_dir}")
    
    # Extract patient ID
    patient_id = os.path.basename(patient_dir)
    
    # Find all visits (folders starting with 'V')
    visit_dirs = [d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d)) and d.startswith('V')]
    visit_dirs.sort()
    
    # Dictionary to store all patient data
    patient_data = {}
    
    # Create output directories
    train_dir = os.path.join(out_dir, "train")
    eval_dir = os.path.join(out_dir, "eval")
    test_dir = os.path.join(out_dir, "test")
    
    for directory in [train_dir, eval_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Track dataset statistics
    X_MIN_ARRAY = []
    X_MAX_ARRAY = []
    Y_UNIQUE_ARRAY = []
    X_LEN_ARRAY = []
    
    # Process each visit for train/eval/test split
    for visit_idx, visit_name in enumerate(visit_dirs):
        visit_path = os.path.join(patient_dir, visit_name)
        patient_data[visit_name] = {}
        
        # Determine whether it should go to train/eval/test
        if visit_idx < len(visit_dirs) * 0.6:
            output_dir = train_dir
        elif visit_idx < len(visit_dirs) * 0.8:
            output_dir = eval_dir
        else:
            output_dir = test_dir
            
        # Find all days (folders starting with 'D')
        day_dirs = [d for d in os.listdir(visit_path) if os.path.isdir(os.path.join(visit_path, d)) and d.startswith('D')]
        day_dirs.sort()
        
        # Process each day
        for day_idx, day_name in enumerate(day_dirs):
            day_path = os.path.join(visit_path, day_name)
            patient_data[visit_name][day_name] = {}
            
            # Check for session types
            for session_type in SESSION_TYPES.keys():
                session_path = os.path.join(day_path, session_type)
                
                if os.path.exists(session_path):
                    patient_data[visit_name][day_name][session_type] = []
                    
                    # Find all .vhdr files in this session
                    vhdr_files = glob(os.path.join(session_path, "**", "*.vhdr"), recursive=True)
                    
                    for vhdr_idx, vhdr_path in enumerate(vhdr_files):
                        # Process the VHDR file
                        eeg_data = process_vhdr_file(vhdr_path)
                        
                        if eeg_data is not None:
                            # Extract features and labels using the HMC-aligned approach
                            features, labels = extract_features_and_labels(eeg_data)
                            
                            if len(features) == 0:
                                continue
                                
                            # Save each segment as a separate sample
                            for idx in range(features.shape[0]):
                                sample = {
                                    "X": features[idx],
                                    "ch_names": eeg_data["Channels"],
                                    "y": int(labels[idx][0]),  # Use the mapped label index
                                    # "label_meaning": label_meaning  # Include meaning dictionary
                                }
                                
                                # Track statistics
                                X_MIN_ARRAY.append(np.min(features[idx]))
                                X_MAX_ARRAY.append(np.max(features[idx]))
                                X_LEN_ARRAY.append(features[idx].shape[1])
                                Y_UNIQUE_ARRAY.append(int(labels[idx][0]))
                                
                                # Save sample
                                out_filename = f"{patient_id}_{visit_name}_{day_name}_{session_type}_{vhdr_idx}_{idx}.pkl"
                                with open(os.path.join(output_dir, out_filename), "wb") as f:
                                    pickle.dump(sample, f)
    
    # Print statistics
    if X_LEN_ARRAY:
        print("Dataset statistics:")
        print(f"Min value: {min(X_MIN_ARRAY)}")
        print(f"Max value: {max(X_MAX_ARRAY)}")
        print(f"Sample lengths: all exactly {X_LEN_ARRAY[0]} time points (30 seconds at 200Hz)")
        print(f"Unique labels: {set(Y_UNIQUE_ARRAY)}")
        print(f"Number of samples: {len(X_LEN_ARRAY)}")
    else:
        print("No samples were processed")
    
    return patient_data

def process_dataset(root_dir, out_dir):
    """Process the entire locked-in state dataset"""
    print(f"Processing locked-in state dataset from {root_dir}")
    
    # Find all patient directories
    patient_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('P')]
    patient_dirs.sort()
    
    for patient_dir in patient_dirs:
        patient_path = os.path.join(root_dir, patient_dir)
        process_patient_data(patient_path, out_dir)
    
    print("Dataset processing completed.")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process locked-in state dataset for NeuroLM')
    
    parser.add_argument('--input', type=str, default=r'D:\datasets\zenodo.org\Raw Files',
                        help='Input directory containing the raw dataset')
                        
    parser.add_argument('--output', type=str, default=r'E:\dataset\LockedIn',
                        help='Output directory for processed data')
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)
