"""
EOG Dataset Preparation Script (for datasets D, E, F)
Based on the NeuroLM framework
"""

import numpy as np
import os
import pickle
from scipy.io import loadmat
from scipy import signal

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

# def test_bandpass_filter():
#     import mne
#     import matplotlib.pyplot as plt
#     # generate a longer test signal (10 seconds instead of 1)
#     fs = 256
#     t = np.arange(0, 10, 1/fs)  # 10 seconds of data
#     x = np.sin(2 * np.pi * 1 * t)  # 1 Hz component
#     x += np.sin(2 * np.pi * 10 * t)  # 10 Hz component
#     x += np.sin(2 * np.pi * 50 * t)  # 50 Hz component (noise)
#     x = np.vstack([x, x])  # Two channels
    
#     # Apply our bandpass filter (0.3-30 Hz)
#     x_bp = bandpass_filter(x, 0.3, 10.0, fs)
    
#     # Apply MNE bandpass filter for comparison
#     x_mne = mne.filter.filter_data(x, fs, 0.3, 10.0, method='fir', fir_design='firwin')
    
#     # Plot results
#     plt.figure(figsize=(12, 8))
    
#     plt.subplot(3, 1, 1)
#     plt.plot(t, x[0])
#     plt.title('Original Signal')
#     plt.xlim(0, 3)  # Show first 3 seconds
    
#     plt.subplot(3, 1, 2)
#     plt.plot(t, x_bp[0])
#     plt.title('Our Bandpass Filter (0.3-30 Hz)')
#     plt.xlim(0, 3)
    
#     plt.subplot(3, 1, 3)
#     plt.plot(t, x_mne[0])
#     plt.title('MNE Bandpass Filter (0.3-30 Hz)')
#     plt.xlim(0, 3)
    
#     plt.tight_layout()
#     plt.show()

# test_bandpass_filter()

def load_eog_data(data_dir, dataset_type):
    """
    Load EOG data from a directory
    dataset_type: 'D', 'E', or 'F' to specify which dataset structure to expect
    """
    subject_dirs = [d for d in os.listdir(data_dir) if d.startswith('S')]
    all_data = []
    
    for subject in subject_dirs:
        subject_path = os.path.join(data_dir, subject)
        
        # Load EOG signals
        eog_path = os.path.join(subject_path, 'EOG.mat')
        eog_data = loadmat(eog_path)['EOG'] if 'EOG' in loadmat(eog_path) else loadmat(eog_path)
        
        # plot eog data
        # import matplotlib.pyplot as plt
        # plt.plot(eog_data.T)
        # plt.show()
        # Load control signal (trial segmentation)
        control_path = os.path.join(subject_path, 'ControlSignal.mat')
        control_data = loadmat(control_path)['ControlSignal'] if 'ControlSignal' in loadmat(control_path) else loadmat(control_path)


        # remove last point of control signal and eog data
        control_data = control_data[:, :-1]
        eog_data = eog_data[:, :-1]

        # plot eog data
        # import matplotlib.pyplot as plt
        # plt.plot(eog_data.T)
        # plt.show()

        # Apply bandpass filter (0.3-30 Hz)
        current_rate = 256  # Original sampling rate
        eog_data = bandpass_filter(eog_data, 0.3, 30.0, current_rate)

        # # plot eog data
        # plt.plot(eog_data.T)
        # plt.show()

        # reshape data
        target_rate = 100
        if current_rate != target_rate:
            # Import necessary module for resampling

            # Calculate new length based on the ratio of target rate to current rate
            new_length = int(eog_data.shape[1] * (target_rate / current_rate))

            # Resample the data
            eog_data = signal.resample(eog_data, new_length, axis=1)
            
            # manually resample control signal
            control_data = control_data[:, np.linspace(0, control_data.shape[1] - 1, new_length, dtype=int)]
        
        
        # Extract trials based on control signal
        trials = []
        for trial_type in [1, 2, 3]:  # 1=forward saccade, 2=return saccade, 3=blink
            # Find indices where control signal equals trial_type
            trial_indices = np.where(control_data == trial_type)[1]
            
            if len(trial_indices) > 0:
                # Group consecutive indices
                breaks = np.where(np.diff(trial_indices) > 1)[0] + 1
                segments = np.split(trial_indices, breaks)
                
                for segment in segments:
                    if len(segment) > 0:
                        start_idx = segment[0]
                        end_idx = segment[-1] + 1
                        
                        # Extract EOG for this segment
                        # After extracting the segment
                        eog_segment = eog_data[:, start_idx:end_idx]
                        
                        remainder = eog_segment.shape[1] - 200
                        if remainder > 0:
                            # truncate to 200n samples
                            eog_segment = eog_segment[:, :200]
                        elif remainder < 0:
                            # Pad with zeros
                            pad_length = - remainder
                            eog_segment = np.pad(eog_segment, ((0, 0), (0, pad_length)), 'constant')
                                                    
                        # Create sample
                        sample = {
                            "X": eog_segment,
                            "y": 1 if trial_type == 1 or trial_type == 2 else 0, # label 1 for saccades, 0 for blinks
                            "ch_names": ["EOGU", "EOGD", "EOGR", "EOGL"] if eog_data.shape[0] == 4 else ['EOG horizontal', 'EOG vertical']
                        }
                        
                        trials.append(sample)
        
        all_data.extend(trials)
    
    return all_data

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

def prepare_eog_dataset(dataset_type, root_dir, out_dir):
    """
    Prepare EOG dataset for training
    dataset_type: 'D', 'E', or 'F'
    """
    print(f"Processing EOG dataset {dataset_type}...")
    
    train_dir = os.path.join(out_dir, "train")
    eval_dir = os.path.join(out_dir, "eval")
    test_dir = os.path.join(out_dir, "test")
    
    samples = load_eog_data(root_dir, dataset_type)
    
    # Split into train/eval/test (80/10/10)
    total_samples = len(samples)
    train_size = int(0.6 * total_samples)
    eval_size = int(0.2 * total_samples)
    
    # Randomize samples
    np.random.seed(42)
    np.random.shuffle(samples)
    
    train_samples = samples[:train_size]
    eval_samples = samples[train_size:train_size+eval_size]
    test_samples = samples[train_size+eval_size:]
    
    save_samples(train_samples, train_dir)
    save_samples(eval_samples, eval_dir)
    save_samples(test_samples, test_dir)
    
    print(f"Dataset {dataset_type} processed: {len(train_samples)} train, {len(eval_samples)} eval, {len(test_samples)} test samples.")

# Main execution
if __name__ == "__main__":
    # Dataset D
    prepare_eog_dataset(
        'D',
        root_dir=r"D:\datasets\www.um.edu.mt\DATASET",
        out_dir=r"E:\dataset\EOG_Dataset_D"
    )
    
    # Dataset E
    prepare_eog_dataset(
        'E',
        root_dir=r"D:\datasets\www.um.edu.mt\Dataset_NonStationary", 
        out_dir=r"E:\dataset\EOG_Dataset_E"
    )
    
    # Dataset F
    prepare_eog_dataset(
        'F',
        root_dir=r"D:\datasets\www.um.edu.mt\Dataset_Stationary",
        out_dir=r"E:\dataset\EOG_Dataset_F"
    )
