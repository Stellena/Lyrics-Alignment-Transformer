import numpy as np
import torch
from torch.utils.data import Dataset


class ScoreDataset(Dataset):
    def __init__(self, input_dir, target_dir, config, transform=None, target_transform=None):
        self.PAD, self.EOS, self.MAXLEN = config
        self.input_data = self.generate_preprocessed_data(input_dir)
        self.target_data = self.generate_preprocessed_data(target_dir)
    
    # npy 파일 경로를 받으면, EOS 넣고 PAD 채워줌으로써 일정한 길이의 텐서로 변환한다.
    def generate_preprocessed_data(self, data_dir):
        data_raw = np.load(data_dir, allow_pickle=True)
        data_len = len(data_raw)
        data = np.full((data_len, self.MAXLEN), self.PAD, dtype=np.int64)
        for i, sample in enumerate(data_raw):
            sample_len = len(sample)
            assert sample_len < 2048
            data[i][1:1+sample_len] = sample
            data[i][1+sample_len] = self.EOS
        return data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]

