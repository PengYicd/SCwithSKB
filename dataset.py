import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import sentence_transformers


def collate_data(batch):
    batch_size = len(batch)
    data_x = torch.zeros([batch_size, 20])
    data_y = torch.zeros([batch_size, 20])
    for i, (x, y) in enumerate(batch):
        x, y =np.array(x), np.array(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        data_x[i] = x
        data_y[i] = y
    
    return data_x.int(), data_y.int()


class SentenceData(Dataset):
    def __init__(self, data_src,  data_tgt):
        self.data_src = data_src
        self.data_tgt = data_tgt

    def sample_loader(self, data):
        return random.shuffle(data)

    def __len__(self):
        return len(self.data_src)

    def __getitem__(self, index):
        return self.data_src[index], self.data_tgt[index]





