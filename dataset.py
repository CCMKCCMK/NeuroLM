"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

from torch.utils.data import Dataset
import torch
from einops import rearrange
import pickle


standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", \
    'pad', 'I1', 'I2', \
    'EOG1', 'EOG2', 'EOG3', 'EOG4', \
    'EOGU', 'EOGD', 'EOGR', 'EOGL', \
    'EOGRU', 'EOGRD','EOGRH', 'EOGLH', 'EOGUR', 'EOGDR', 'EOGUL', 'EOGDL', 'EOGDiagR', 'EOGDiagLU',
    'EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal',
    'E1', 'E2',
    'EOG horizontal', 'EOG vertical'
]


class PickleLoader(Dataset):
    def __init__(self, files, block_size=1024, sampling_rate=100, GPT_training=False):
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.GPT_training = GPT_training

    def __len__(self):
        return len(self.files)
    
    def std_norm(self, x):
            mean = torch.mean(x, dim=(0, 1), keepdim=True)
            std = torch.std(x, dim=(0, 1), keepdim=True)
            x = (x - mean) / std
            return x

    def get_chans(self, ch_names):
            chans = []
            for ch_name in ch_names:
                chans.append(standard_1020.index(ch_name))
            return chans

    def __getitem__(self, index):
        sample = pickle.load(open(self.files[index], "rb"))
        data = sample["X"]
        ch_names = sample["ch_names"]
        data = torch.FloatTensor(data / 100)

        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]

        data = rearrange(data, 'N (A T) -> (A N) T', T=200)
        
        X = torch.zeros((self.block_size, 200))
        X[:data.size(0)] = data

        if not self.GPT_training:
            Y_freq = torch.zeros((self.block_size, 100))
            Y_raw = torch.zeros((self.block_size, 200))
            x_fft = torch.fft.fft(data, dim=-1)
            amplitude = torch.abs(x_fft)
            amplitude = self.std_norm(amplitude)
            Y_freq[:data.size(0)] = amplitude[:, :100]
            Y_raw[:data.size(0)] = self.std_norm(data)
        
        # input_chans is the indices of the channels in the standard_1020 list
        # used for the spatial embedding
        input_chans = list(ch_names) * time
        input_chans.extend(['pad'] * (self.block_size - data.size(0)))
        input_chans = torch.IntTensor(self.get_chans(input_chans))
        # input_time is the mask for padding zeros
        # ensure that padding zeros are not used in the attention mechanism
        input_time.extend([0] * (self.block_size - data.size(0)))
        input_time = torch.IntTensor(input_time)

        input_mask = torch.ones(self.block_size)
        input_mask[data.size(0):] = 0

        if self.GPT_training:
            # gpt_mask is the mask for the GPT model
            gpt_mask = torch.tril(torch.ones(self.block_size, self.block_size)).view(1, self.block_size, self.block_size)
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1

            gpt_mask[:, :, num_chans * time:] = 0
            return X, input_chans, input_time, input_mask.bool(), gpt_mask.bool(), num_chans, data.size(0)
        
        return X, Y_freq, Y_raw, input_chans, input_time, input_mask.bool()
