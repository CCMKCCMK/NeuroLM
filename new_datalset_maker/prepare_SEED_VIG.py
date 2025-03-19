"""
EOG and EEG Dataset Preparation Script (for datasets SEED VIG)
Based on the NeuroLM framework
"""

import numpy as np
import os
import pickle
from scipy.io import loadmat
from scipy import signal
# from scipy.fft import fft, fftfreq

# Taken from Ifeachor and Jervis p. 356.
# Note that here the passband ripple and stopband attenuation are
# rendundant. The scalar passband ripple δp is expressed in dB as
# 20 * log10(1+δp), but the scalar stopband ripple δs is expressed in dB as
# -20 * log10(δs). So if we know that our stopband attenuation is 53 dB
# (Hamming) then δs = 10 ** (53 / -20.), which means that the passband
# deviation should be 20 * np.log10(1 + 10 ** (53 / -20.)) == 0.0194.
_fir_window_dict = {
    "hann": dict(name="Hann", ripple=0.0546, attenuation=44),
    "hamming": dict(name="Hamming", ripple=0.0194, attenuation=53),
    "blackman": dict(name="Blackman", ripple=0.0017, attenuation=74),
}

def bandpass_filter(data, lowcut, highcut, fs, filter_length='auto', fir_window='hamming'):
    """
    Apply a FIR bandpass filter to the input data using sophisticated design method
    
    Parameters:
    data -- data to filter (channels x samples)
    lowcut -- low cutoff frequency (high-pass edge)
    highcut -- high cutoff frequency (low-pass edge)
    fs -- sampling rate
    filter_length -- length of the FIR filter or 'auto' for automatic selection
    fir_window -- window function to use (default: 'hamming')
    """
    # Get window characteristics
    if fir_window in _fir_window_dict:
        window_info = _fir_window_dict[fir_window]
        print(f"Using {window_info['name']} window with {window_info['ripple']:0.4f} passband ripple "
              f"and {window_info['attenuation']:d} dB stopband attenuation")
    
    # Calculate transition bandwidths
    nyq = fs / 2.0
    
    # Better transition bandwidth calculation (from provided code)
    l_trans_bandwidth = np.minimum(np.maximum(0.25 * lowcut, 2.0), lowcut)
    h_trans_bandwidth = np.minimum(np.maximum(0.25 * highcut, 2.0), nyq - highcut)
    
    # Calculate stop frequencies
    l_stop = lowcut - l_trans_bandwidth  # Lower stop frequency
    h_stop = highcut + h_trans_bandwidth  # Upper stop frequency
    
    # Check for valid frequencies
    if l_stop < 0:
        l_stop = 0
    if h_stop > nyq:
        h_stop = nyq
        
    # Determine filter length if auto
    if filter_length == 'auto':
        # Adjust length factor based on window type for optimal response
        length_factor = 3.3  # Default for hamming
        if fir_window == 'hann':
            length_factor = 3.1
        elif fir_window == 'blackman':
            length_factor = 5.0
            
        min_trans_bandwidth = min(l_trans_bandwidth, h_trans_bandwidth)
        filter_length = int(np.ceil(length_factor * fs / min_trans_bandwidth))
        
        # Ensure odd length for linear phase
        filter_length += (filter_length % 2 == 0)
        
        # Ensure filter length isn't too long for the data
        max_length = data.shape[1] // 3  # Maximum 1/3 of data length
        if filter_length > max_length and max_length > 0:
            print(f"Warning: Reducing filter length from {filter_length} to {max_length} due to short signal")
            filter_length = max_length
            # Ensure odd length for linear phase
            filter_length += (filter_length % 2 == 0)
    
    print(f"Filter length: {filter_length} samples ({filter_length / fs:0.3f} s)")
    print(f"Bandpass: {lowcut:.2f}-{highcut:.2f} Hz")
    print(f"Lower transition bandwidth: {l_trans_bandwidth:.2f} Hz")
    print(f"Upper transition bandwidth: {h_trans_bandwidth:.2f} Hz")
    
    # Build frequency response for firwin2 - avoid duplicate frequencies
    if l_stop == 0:
        # If l_stop is 0, start directly with lowcut to avoid duplicate 0 values
        freq = [0, lowcut, highcut, h_stop, nyq]
        gain = [0, 1, 1, 0, 0]
    else:
        freq = [0, l_stop, lowcut, highcut, h_stop, nyq]
        gain = [0, 0, 1, 1, 0, 0]
    
    # Design the FIR filter
    h = signal.firwin2(filter_length, freq, gain, fs=fs, window=fir_window)
    
    # Apply filter to each channel using zero-phase filtering
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        # Use a padding length that's appropriate for the signal length
        # Default padlen in filtfilt is 3 * (filter_length - 1) // 2
        padlen = min(data.shape[1] - 1, 3 * (filter_length - 1) // 2)
        filtered_data[i] = signal.filtfilt(h, [1.0], data[i], padlen=padlen)
    
    return filtered_data

def notch_filter(data, freq, fs, q=30.0):
    """
    Apply a notch filter to remove power line noise
    
    Parameters:
    data -- data to filter (channels x samples)
    freq -- frequency to remove (Hz)
    fs -- sampling rate (Hz)
    q -- quality factor
    """
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = signal.iirnotch(w0, q)
    
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, data[i])
    
    return filtered_data

def save_samples(samples, out_dir):
    """Save samples as pickle files"""
    os.makedirs(out_dir, exist_ok=True)
    
    for i, sample in enumerate(samples):
        filename = os.path.join(
            out_dir, 
            f"{i}.pkl"
        )
        
        with open(filename, "wb") as f:
            pickle.dump(sample, f)

def load_seed_vig_data(data_dir, perclos_dir):
    """
    Load SEED VIG EOG and EEG data and PERCLOS labels
    
    Parameters:
    data_dir -- path to Raw_Data folder containing EOG and EEG data
    perclos_dir -- path to perclos_labels folder containing PERCLOS indices
    """
    all_samples = []
    
    # Get all subject files
    subject_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    
    for subject_file in subject_files:
        subject_path = os.path.join(data_dir, subject_file)
        subject_id = os.path.splitext(subject_file)[0]
        
        # Load data
        print(f"Loading data for subject {subject_id}...")
        data = loadmat(subject_path)
        
        eog_data = data['EOG']
        eeg_data = data['EEG']
        
        # Extract horizontal and vertical EOG channels
        eog_h = eog_data['eog_h'][0][0]  # Horizontal EOG
        eog_v = eog_data['eog_v'][0][0]  # Vertical EOG
        eog_channels = np.vstack((eog_h.T, eog_v.T))# Stack into channel x samples format
        eog_ch_names = ['EOG horizontal', 'EOG vertical']

        # Extract EEG channels
        eeg_channels = eeg_data['data'][0][0].T  # Convert to channels x samples
        eeg_ch_names = ['EEG ' + ch[0] for ch in eeg_data['chn'][0][0][0]]
                       
        eog_rate = 125
        eeg_rate = 200

        # Pre-processing: Apply bandpass filter for EOG (0.3-30 Hz)
        eog_filtered = bandpass_filter(eog_channels, 0.3, 30.0, eog_rate)
        # Pre-processing: Apply bandpass filter for EEG (0.1-75 Hz)
        eeg_processed = bandpass_filter(eeg_channels, 0.1, 75.0, eeg_rate)        

        # Upsampling EOG to 200 Hz
        target_rate = 200
        print(f"Upsampling EOG from {eog_rate}Hz to {target_rate}Hz...")
        new_length = int(eog_filtered.shape[1] * (target_rate / eog_rate))
        eog_processed = signal.resample(eog_filtered, new_length, axis=1)


        # Apply notch filter at 50Hz to remove power line noise ALREADY DONE IN original data
        # Resample EEG to 200 Hz ALREADY DONE IN original data

        # Load PERCLOS labels
        perclos_file = os.path.join(perclos_dir, f"{subject_id}.mat")
        if not os.path.exists(perclos_file):
            print(f"Warning: PERCLOS file not found for {subject_id}, skipping")
            raise FileNotFoundError(f"PERCLOS file not found for {subject_id}")
            
        perclos_data = loadmat(perclos_file)['perclos']
        
        # Number of segments in PERCLOS (typically 885 for SEED-VIG)
        num_segments = perclos_data.shape[0]
        print(f"Number of segments with PERCLOS labels: {num_segments}")
        
        # Calculate samples per segment
        samples_per_segment = int(eog_processed.shape[1] / num_segments)
        
        # Process each segment with its PERCLOS label
        for i in range(num_segments):

            start = i * samples_per_segment
            end = min((i + 1) * samples_per_segment, eog_processed.shape[1])
            segment_eog = eog_processed[:, start:end]
            segment_eeg = eeg_processed[:, start:end]

            # Get PERCLOS value and determine vigilance state
            perclos_value = perclos_data[i][0]
            
            # Categorize based on PERCLOS thresholds
            if perclos_value < 0.35:
                label = 0  # awake
            elif perclos_value < 0.7:
                label = 1  # tired
            else:
                label = 2  # drowsy

            # Combine EOG and EEG data
            final_signal = np.concatenate((segment_eog, segment_eeg), axis=0)
            final_ch_names = eog_ch_names + eeg_ch_names
            # Create sample dictionary with both EOG and EEG data
            sample = {
                "X": final_signal,
                "ch_names": final_ch_names,
                "y": label,
                "subject": subject_id,
            }

            all_samples.append(sample)
    
    print(f"Total samples collected: {len(all_samples)}")
    return all_samples

def prepare_seed_vig_dataset(root_dir, perclos_dir, out_dir):
    """
    Prepare SEED VIG dataset for training
    
    Parameters:
    root_dir -- path to Raw_Data directory
    perclos_dir -- path to perclos_labels directory
    out_dir -- path to output directory
    """
    print("Processing SEED VIG dataset...")
    
    train_dir = os.path.join(out_dir, "train")
    eval_dir = os.path.join(out_dir, "eval")
    test_dir = os.path.join(out_dir, "test")
    
    samples = load_seed_vig_data(root_dir, perclos_dir)
    
    # Get subject IDs for stratified splitting
    subject_ids = list(set([s["subject"] for s in samples]))
    print(f"Found {len(subject_ids)} subjects")
    
    # Shuffle subjects for random split
    np.random.seed(42)
    np.random.shuffle(subject_ids)
    
    # Split subjects into train/eval/test (60/20/20)
    train_subjects = subject_ids[:int(0.6 * len(subject_ids))]
    eval_subjects = subject_ids[int(0.6 * len(subject_ids)):int(0.8 * len(subject_ids))]
    test_subjects = subject_ids[int(0.8 * len(subject_ids)):]
    
    # Assign samples to splits based on subject
    train_samples = [s for s in samples if s["subject"] in train_subjects]
    eval_samples = [s for s in samples if s["subject"] in eval_subjects]
    test_samples = [s for s in samples if s["subject"] in test_subjects]
    
    save_samples(train_samples, train_dir)
    save_samples(eval_samples, eval_dir)
    save_samples(test_samples, test_dir)
    
    print(f"SEED VIG dataset processed: {len(train_samples)} train, {len(eval_samples)} eval, {len(test_samples)} test samples.")
    print(f"Train subjects: {train_subjects}")
    print(f"Eval subjects: {eval_subjects}")
    print(f"Test subjects: {test_subjects}")

# Main execution
if __name__ == "__main__":
    # SEED VIG Dataset
    prepare_seed_vig_dataset(
        root_dir=r"D:\datasets\SEED-VIG\Raw_Data",
        perclos_dir=r"D:\datasets\SEED-VIG\perclos_labels",
        out_dir=r"E:\dataset\SEED-VIG"
    )
