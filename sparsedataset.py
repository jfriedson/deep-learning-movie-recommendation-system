import copy

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
import torch
from torch.utils.data import Dataset

class SparseDataset(Dataset):
    def __init__(self, file, n_movies):
        self.sparse_mat = load_npz(file)
        self.n_users = self.sparse_mat.shape[0]
        self.n_movies = n_movies
        self.targets_percent = .1

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        target = self.sparse_mat.getrow(idx).A
        target.resize(self.n_movies)
        
        idxs = target.nonzero()[0]
        idxs = idxs[np.random.rand(len(idxs)) < self.targets_percent]

        obs = copy.deepcopy(target)
        obs[idxs] = 0.

        return obs, target
