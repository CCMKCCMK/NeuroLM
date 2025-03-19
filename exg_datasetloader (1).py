import copy

from torch.utils.data import Dataset
from pathlib import Path
import h5py
import bisect
import torch
from einops import rearrange
import tiktoken
import numpy as np
import pickle
import os

from llamadatasets.dataset import standard_1020

def get_chans(ch_names):
    chans = []
    for ch_name in ch_names:
        chans.append(standard_1020.index(ch_name))
    return chans

class HMCLoader(Dataset):
    def __init__(self, dataset_config, tokenizer, train_config, partition="train", sampling_rate=200, is_instruct=False,
                 is_val=False):
        # self.root = dataset_config.train_path
        if partition == "train":
            files = Path(dataset_config.train_path).rglob('*.pkl')
            self.is_instruct = False
        elif partition == "eval":
            files = Path(dataset_config.val_path).rglob('*.pkl')
            self.is_instruct = True
        elif partition == "test":
            files = Path(dataset_config.test_path).rglob('*.pkl')
            self.is_instruct = True
        self.dataset_name = dataset_config.dataset
        self.files = [file for file in files]
        # self.files = self.files[:2000]
        self.tokenizer = tokenizer
        # self.tokenizer.pad_token_id = 50256
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.signal_max_len = train_config.context_length
        self.text_max_len = train_config.text_length
        self.text = {
        0:'(A)',
        1:'(B)',
        2:'(C)',
        3:'(D)',
        4:'(E)',
        }
        self.prompt = 'Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer:'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        IGNORE_INDEX = -100
        sample = pickle.load(open(self.files[index], "rb"))
        X = sample["X"] # 电信号数据
        Y = int(sample["y"]) # 电信号label
        # 编码prompt和answer，并且拼接为整体的example
        # example = self.prompt + self.text[Y]
        prompt = self.prompt
        question = self.tokenizer.encode(prompt)
        answer = self.tokenizer.encode(self.text[Y], add_special_tokens=False)
        prompt = torch.tensor(question, dtype=torch.int64)
        example = question + answer + [self.tokenizer.eos_token_id]
        if self.is_instruct:
            question = torch.tensor(question, dtype=torch.int64)
            valid_question_len = question.size(0)
            if self.text_max_len > valid_question_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_question_len] = copy.deepcopy(question)
                question = text_pad
            question_mask = question.ge(0)

            question_mask &= question.ne(50256)

        # example = self.tokenizer.encode(example)
        # example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        original_example_length = len(example)
        # 设置example和label的mask
        valid_text_len = example.size(0)
        if self.text_max_len > valid_text_len:
            text_pad = torch.full((self.text_max_len,), fill_value=50256)
            text_pad[:valid_text_len] =copy.deepcopy(example)
            example = text_pad

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        labels[original_example_length:] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)

        example_mask &= example.ne(50256)

        # example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)
        ch_names = sample["ch_names"]
        input_chans = list(ch_names) * time

        # pad sinal to eeg_max_len
        valid_signal_len = data.size(0)
        if self.signal_max_len > data.size(0):
            signal = torch.zeros((self.signal_max_len, 200))
            signal[:data.size(0)] = data
            signal_mask = torch.ones(self.signal_max_len)
            signal_mask[valid_signal_len:] = 0

            input_chans.extend(['pad'] * (self.signal_max_len - data.size(0)))
            input_time.extend([0] * (self.signal_max_len - data.size(0)))
        else:
            signal = data
            signal_mask = torch.ones(data.size(0), dtype=torch.bool)

        signal_labels_mask = torch.zeros(signal.size(0), dtype=torch.bool)
        signal_labels = torch.zeros(signal.size(0), dtype=torch.int64)
        signal_labels[~signal_labels_mask] = IGNORE_INDEX


        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        if self.is_instruct:
            return {"prompt": question.tolist(),
                    "prompt_mask": question_mask.tolist(),
                    "input_ids": example.tolist(),
                    "labels": labels.tolist(),
                    "attention_mask": example_mask.tolist(),
                    "signal": signal.tolist(),
                    "input_chans": input_chans.tolist(),
                    "input_time": input_time.tolist(),
                    "signal_mask": signal_mask.tolist(),
                    "signal_labels": signal_labels.tolist(),
                    "target_text": Y
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
                    "target_text": Y
                    }