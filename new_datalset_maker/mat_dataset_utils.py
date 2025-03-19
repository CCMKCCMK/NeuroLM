"""
Unified utility functions for EEG/EOG dataset processing
"""

import numpy as np
from scipy import signal
import mne
import os
import pickle

def bandpass_filter(data, lowcut, highcut, fs, filter_length='auto', fir_window='hamming'):
    """
    Apply a FIR bandpass filter to the input data
    
    Parameters:
    data -- data to filter (channels x samples)
    lowcut -- low cutoff frequency (high-pass edge)
    highcut -- high cutoff frequency (low-pass edge)
    fs -- sampling rate
    filter_length -- length of the FIR filter or 'auto' for automatic selection
    fir_window -- window function to use (default: 'hamming')
    """
    nyq = fs / 2.0
    
    # Calculate transition bandwidths
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
        # Default length factor for hamming window
        length_factor = 3.3
        
        min_trans_bandwidth = min(l_trans_bandwidth, h_trans_bandwidth)
        filter_length = int(np.ceil(length_factor * fs / min_trans_bandwidth))
        
        # Ensure odd length for linear phase
        filter_length += (filter_length % 2 == 0)
        
        # Ensure filter length isn't too long for the data
        max_length = data.shape[1] // 3  # Maximum 1/3 of data length
        if filter_length > max_length and max_length > 0:
            print(f"Warning: Reducing filter length from {filter_length} to {max_length}")
            filter_length = max_length
            # Ensure odd length for linear phase
            filter_length += (filter_length % 2 == 0)
    
    # Build frequency response for firwin2
    if l_stop == 0:
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

def resample_signal(data, current_fs, target_fs):
    """
    Resample signal from current_fs to target_fs
    
    Parameters:
    data -- data to resample (channels x samples)
    current_fs -- current sampling frequency
    target_fs -- target sampling frequency
    """
    if current_fs == target_fs:
        return data
    
    # Calculate new length
    new_length = int(data.shape[1] * (target_fs / current_fs))
    
    # Resample
    resampled_data = np.zeros((data.shape[0], new_length))
    for i in range(data.shape[0]):
        resampled_data[i] = signal.resample(data[i], new_length)
    
    return resampled_data

def process_signals(signals, ch_names, current_fs=None, target_fs=200):
    """
    Process signals by applying appropriate filters based on channel types
    
    Parameters:
    signals -- signal data (channels x samples)
    ch_names -- list of channel names
    current_fs -- current sampling rate (if None, assumes it's already at target_fs)
    target_fs -- target sampling rate
    """
    # Identify channel types
    eog_indices = [i for i, ch in enumerate(ch_names) if 'EOG' in ch]
    eeg_indices = [i for i, ch in enumerate(ch_names) if 'EEG' in ch]
    
    # Process EOG channels (0.3-30 Hz)
    if eog_indices:
        eog_signals = signals[eog_indices]
        eog_filtered = bandpass_filter(eog_signals, 0.3, 30.0, target_fs)
        for i, idx in enumerate(eog_indices):
            signals[idx] = eog_filtered[i]
    
    # Process EEG channels (0.1-75 Hz with 50Hz notch)
    if eeg_indices:
        eeg_signals = signals[eeg_indices]
        eeg_filtered = bandpass_filter(eeg_signals, 0.1, 75.0, target_fs)
        eeg_notched = notch_filter(eeg_filtered, 50.0, target_fs)
        for i, idx in enumerate(eeg_indices):
            signals[idx] = eeg_notched[i]
    
    # Resample if needed
    if current_fs is not None and current_fs != target_fs:
        signals = resample_signal(signals, current_fs, target_fs)
    
    return signals

def save_pickle(obj, filename):
    """Save object as pickle file"""
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
