import copy
import torch
from torch.utils.data import Dataset
from pathlib import Path
import bisect
from einops import rearrange
import numpy as np
import pickle
import os

# Updated standard 1020 channels to include EOG channels
standard_1020 = [
    'EEG Fpz-Cz', 'EEG Pz-Oz',
    'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2',
    'EOG E1-M2', 'EOG E2-M2',
    'EOG E1', 'EOG E2', 
    'EEG F3', 'EEG C3', 'EEG O1', 'EEG F4', 'EEG C4', 'EEG O2',
    'EOG horizontal', 'EOG vertical', 'pad'
]

def get_chans(ch_names):
    """Map channel names to indices in standard_1020 list"""
    chans = []
    for ch_name in ch_names:
        print(f"Channel name: {ch_name}")
        chans.append(standard_1020.index(ch_name))
    return chans

class UnifiedEXGLoader(Dataset):
    """
    Unified dataloader for EEG and EOG signals across different datasets
    
    Supports:
    - HMC (sleep stages)
    - SleepEDF (sleep stages) 
    - SEED-VIG (vigilance states)
    - ISRUC-SLEEP (sleep stages)
    """
    def __init__(self, dataset_config, tokenizer, train_config, partition="train", sampling_rate=200, is_instruct=False, 
                 is_val=False, signal_types='EOG'):
        """
        Initialize the dataloader
        
        Args:
            dataset_config: Configuration for the dataset
            tokenizer: Tokenizer for text encoding
            train_config: Training configuration
            partition: Data partition ('train', 'eval', or 'test')
            sampling_rate: Target sampling rate (default: 200Hz)
            is_instruct: Whether to use instruction format for evaluation
            signal_types: List of signal types to include ('EEG' or 'EOG' or 'EEG,EOG')
        """
        # Set dataset paths based on partition
        if partition == "train":
            files = Path(dataset_config.train_path).rglob('*.pkl')
            self.is_instruct = False if not is_instruct else True
        elif partition == "eval":
            files = Path(dataset_config.val_path).rglob('*.pkl')
            self.is_instruct = True
        elif partition == "test":
            files = Path(dataset_config.test_path).rglob('*.pkl')
            self.is_instruct = True
        
        # Set signal types to use (default: use all available signals)
        self.signal_types = signal_types.split(',')
        print(f"Using signal types: {self.signal_types}")
        
        # Store dataset name for format-specific processing
        self.dataset_name = dataset_config.dataset
        print(f"Dataset: {self.dataset_name}")
        
        # Load all files
        self.files = [file for file in files]
        self.tokenizer = tokenizer
        self.default_rate = sampling_rate
        self.sampling_rate = sampling_rate
        self.signal_max_len = train_config.context_length
        self.text_max_len = train_config.text_length
        
        # Set prompts and labels based on dataset type
        if self.dataset_name in ["HMC", "SleepEDF", "ISRUC-SLEEP"]:
            # Sleep stage classification
            self.text = {
                0: '(A)',  # Wake
                1: '(B)',  # NREM-1
                2: '(C)',  # NREM-2
                3: '(D)',  # NREM-3
                4: '(E)',  # REM
            }
            signal_type_str = ','.join(self.signal_types)
            self.prompt = f'Question: Which sleep type does this {signal_type_str} segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer:'
        
        elif self.dataset_name == "SEED-VIG":
            # Vigilance state classification
            self.text = {
                0: '(A)',  # Awake
                1: '(B)',  # Tired
                2: '(C)',  # Drowsy
            }
            signal_type_str = ','.join(self.signal_types)
            self.prompt = f'Question: What is the vigilance state in this {signal_type_str} segment? Options: (A) Awake. (B) Tired. (C) Drowsy. Answer:'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        IGNORE_INDEX = -100
        try:
            with open(self.files[index], "rb") as f:
                sample = pickle.load(f)
        except Exception as e:
            print(f"Error loading file {self.files[index]}: {e}")
            # Try next file if available
            if index < len(self.files) - 1:
                return self.__getitem__(index + 1)
            else:
                print("No more files available")
                # Return a placeholder empty sample
                return self.__getitem__(0)  # Recursively call with first index as fallback
        
        # These datasets store data in "X" key
        signal_data = sample["X"]
        label = int(sample["y"])
        ch_names = sample["ch_names"]
        
        # Check which signals we want to include
        # Filter channels based on signal type
        eeg_indices = [i for i, ch in enumerate(ch_names) if 'EEG' in ch]
        eog_indices = [i for i, ch in enumerate(ch_names) if 'EOG' in ch]
        
        # Determine which channels to keep based on requested signal types
        keep_indices = []
        if "EEG" in self.signal_types:
            keep_indices.extend(eeg_indices)
        if "EOG" in self.signal_types:
            keep_indices.extend(eog_indices)
        
        # If no matching channels found, use all channels
        if not keep_indices:
            print(f"Warning: No channels match requested types {self.signal_types} in file {self.files[index]}. Using all channels.")
            keep_indices = list(range(signal_data.shape[0]))
        
        signal_data = signal_data[keep_indices]
        ch_names = [ch_names[i] for i in keep_indices]
        
        # Set up prompt and answer tokens
        prompt = self.prompt
        question = self.tokenizer.encode(prompt)
        answer = self.tokenizer.encode(self.text[label], add_special_tokens=False)
        prompt_tensor = torch.tensor(question, dtype=torch.int64)
        example = question + answer + [self.tokenizer.eos_token_id]
        
        # Process prompt for instruction learning
        if self.is_instruct:
            question_tensor = torch.tensor(question, dtype=torch.int64)
            valid_question_len = question_tensor.size(0)
            if self.text_max_len > valid_question_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_question_len] = copy.deepcopy(question_tensor)
                question_tensor = text_pad
            question_mask = question_tensor.ge(0)
            question_mask &= question_tensor.ne(50256)

        # Process complete example
        example = torch.tensor(example, dtype=torch.int64)
        original_example_length = len(example)
        
        # Handle text padding and masking
        valid_text_len = example.size(0)
        if self.text_max_len > valid_text_len:
            text_pad = torch.full((self.text_max_len,), fill_value=50256)
            text_pad[:valid_text_len] = copy.deepcopy(example)
            example = text_pad

        labels = copy.deepcopy(example)
        labels[: len(prompt_tensor)] = -1
        labels[original_example_length:] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example_mask &= example.ne(50256)
        labels[~label_mask] = IGNORE_INDEX

        # Process signal data
        data = torch.FloatTensor(signal_data / 100)
        time = data.size(1) // self.sampling_rate
        input_time = [i for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=self.sampling_rate)
        input_chans = list(ch_names) * time

        # Pad signal to max length if needed
        valid_signal_len = data.size(0)
        if self.signal_max_len > data.size(0):
            signal = torch.zeros((self.signal_max_len, self.sampling_rate))
            signal[:data.size(0)] = data
            signal_mask = torch.ones(self.signal_max_len)
            signal_mask[valid_signal_len:] = 0

            input_chans.extend(['pad'] * (self.signal_max_len - data.size(0)))
            input_time.extend([0] * (self.signal_max_len - data.size(0)))
        else:
            # Truncate if signal is too long
            signal = data[:self.signal_max_len]
            signal_mask = torch.ones(self.signal_max_len, dtype=torch.bool)
            input_chans = input_chans[:self.signal_max_len]
            input_time = input_time[:self.signal_max_len]

        signal_labels_mask = torch.zeros(signal.size(0), dtype=torch.bool)
        signal_labels = torch.zeros(signal.size(0), dtype=torch.int64)
        signal_labels[~signal_labels_mask] = IGNORE_INDEX

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        if self.is_instruct:
            return {"prompt": question_tensor.tolist(),
                    "prompt_mask": question_mask.tolist(),
                    "input_ids": example.tolist(),
                    "labels": labels.tolist(),
                    "attention_mask": example_mask.tolist(),
                    "signal": signal.tolist(),
                    "input_chans": input_chans.tolist(),
                    "input_time": input_time.tolist(),
                    "signal_mask": signal_mask.tolist(),
                    "signal_labels": signal_labels.tolist(),
                    "target_text": label,
                    }
        else:
            return {"input_ids": example.tolist(),
                    "labels": labels.tolist(),
                    "attention_mask": example_mask.tolist(),
                    "signal": signal.tolist(),
                    "input_chans": input_chans.tolist(),
                    "input_time": input_time.tolist(),
                    "signal_mask": signal_mask.tolist(),
                    "signal_labels": signal_labels.tolist(),
                    "target_text": label,
                    }

# Dataset-specific loaders (for backward compatibility)
class HMCEXGLoader(UnifiedEXGLoader):
    def __init__(self, dataset_config, tokenizer, train_config, partition="train", 
                 sampling_rate=200, is_instruct=False, is_val=False, signal_types='EOG'):
        super().__init__(dataset_config, tokenizer, train_config, partition,
                         sampling_rate, is_instruct, is_val, signal_types)

class SLEEPEXGLoader(UnifiedEXGLoader):
    def __init__(self, dataset_config, tokenizer, train_config, partition="train", 
                 sampling_rate=200, is_instruct=False, is_val=False, signal_types='EOG'):
        super().__init__(dataset_config, tokenizer, train_config, partition,
                         sampling_rate, is_instruct, is_val, signal_types)
        
class SEEDVIGEXGLoader(UnifiedEXGLoader):
    def __init__(self, dataset_config, tokenizer, train_config, partition="train", 
                 sampling_rate=200, is_instruct=False, is_val=False, signal_types='EOG'):
        super().__init__(dataset_config, tokenizer, train_config, partition,
                         sampling_rate, is_instruct, is_val, signal_types)

class ISRUCEXGLoader(UnifiedEXGLoader):
    def __init__(self, dataset_config, tokenizer, train_config, partition="train", 
                 sampling_rate=200, is_instruct=False, is_val=False, signal_types='EOG'):
        # Set the dataset name explicitly to ensure correct prompts
        dataset_config.dataset = "ISRUC-SLEEP"
        super().__init__(dataset_config, tokenizer, train_config, partition,
                         sampling_rate, is_instruct, is_val, signal_types)

# Factory function to create the appropriate loader based on dataset name
def create_exg_loader(dataset_name, dataset_config, tokenizer, train_config, partition="train", 
                      sampling_rate=200, is_instruct=False, is_val=False, signal_types='EOG'):
    """
    Factory function to create the appropriate dataset loader based on dataset name
    
    Args:
        dataset_name: Name of the dataset ('HMC', 'SleepEDF', 'SEED-VIG', 'ISRUC-SLEEP')
        dataset_config: Configuration for the dataset
        tokenizer: Tokenizer for text encoding
        train_config: Training configuration
        partition: Data partition ('train', 'eval', or 'test')
        sampling_rate: Target sampling rate (default: 200Hz)
        is_instruct: Whether to use instruction format for evaluation
        signal_types: List of signal types to include ('EEG' or 'EOG' or 'EEG,EOG')
    
    Returns:
        An instance of the appropriate dataset loader
    """
    dataset_map = {
        "HMC": HMCEXGLoader,
        "SleepEDF": SLEEPEXGLoader,
        "SEED-VIG": SEEDVIGEXGLoader,
        "ISRUC-SLEEP": ISRUCEXGLoader,
    }
    
    if dataset_name not in dataset_map:
        print(f"Warning: Unknown dataset '{dataset_name}'. Using UnifiedEXGLoader as fallback.")
        return UnifiedEXGLoader(dataset_config, tokenizer, train_config, partition,
                               sampling_rate, is_instruct, is_val, signal_types)
    
    return dataset_map[dataset_name](dataset_config, tokenizer, train_config, partition,
                                     sampling_rate, is_instruct, is_val, signal_types)


if __name__ == "__main__":
    # Create a simple mock configuration objects
    class DatasetConfig:
        def __init__(self, dataset, train_path, val_path, test_path):
            self.dataset = dataset
            self.train_path = train_path
            self.val_path = val_path
            self.test_path = test_path
    
    class TrainConfig:
        def __init__(self, context_length, text_length):
            self.context_length = context_length
            self.text_length = text_length
    
    # Simple mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 50256
        
        def encode(self, text, add_special_tokens=True):
            # Just return some dummy token IDs
            return [100, 101, 102] if text else []
    
    # Test with multiple datasets
    datasets = [
        # SleepEDF dataset
        {
            "name": "SleepEDF",
            "config": DatasetConfig(
                dataset="SleepEDF",
                train_path=r'E:\dataset\SleepEDF_new\sleep-telemetry\train',
                val_path=r'E:\dataset\SleepEDF_new\sleep-telemetry\eval',
                test_path=r'E:\dataset\SleepEDF_new\sleep-telemetry\test'
            )
        },
        # HMC dataset
        {
            "name": "HMC",
            "config": DatasetConfig(
                dataset="HMC",
                train_path=r'E:\dataset\HMC_new\train',
                val_path=r'E:\dataset\HMC_new\eval',
                test_path=r'E:\dataset\HMC_new\test'
            )
        },
        # SEED-VIG dataset
        {
            "name": "SEED-VIG",
            "config": DatasetConfig(
                dataset="SEED-VIG",
                train_path=r'E:\dataset\SEED-VIG\train',
                val_path=r'E:\dataset\SEED-VIG\eval',
                test_path=r'E:\dataset\SEED-VIG\test'
            )
        },
        # ISRUC-SLEEP dataset
        {
            "name": "ISRUC-SLEEP",
            "config": DatasetConfig(
                dataset="ISRUC-SLEEP",
                train_path=r'E:\dataset\ISRUC\train',
                val_path=r'E:\dataset\ISRUC\eval',
                test_path=r'E:\dataset\ISRUC\test'
            )
        }
    ]
    
    train_config = TrainConfig(
        context_length=256,
        text_length=128
    )
    
    tokenizer = MockTokenizer()
    
    # Test each dataset
    for dataset_info in datasets:
        try:
            print(f"\nTesting {dataset_info['name']} dataset...")
            # Try to load the dataset
            dataset = create_exg_loader(
                dataset_name=dataset_info['name'],
                dataset_config=dataset_info['config'], 
                tokenizer=tokenizer,
                train_config=train_config,
                partition="train",
                signal_types="EOG"
            )
            print(f"Successfully initialized dataset with {len(dataset)} samples")
            
            # Try to get a single sample if dataset is not empty
            if len(dataset) > 0:
                try:
                    sample = dataset[0]
                    print(f"Sample keys: {sample.keys()}")
                    print(f"Signal shape: {len(sample['signal'])}x{len(sample['signal'][0])}")
                    print(f"Target label: {sample['target_text']}")
                except Exception as e:
                    print(f"Error accessing sample: {e}")
            else:
                print("Dataset is empty. Make sure sample data exists in the specified path.")
        except Exception as e:
            print(f"Error testing {dataset_info['name']} dataset: {e}")
