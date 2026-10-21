import numpy as np
import torch
from typing import Union
from scipy.sparse import coo_matrix,csr_matrix
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
class SparseDataset():
    """
    Custom Dataset class for scipy sparse matrix

    Adapted from https://discuss.pytorch.org/t/dataloader-loads-data-very-slow-on-sparse-tensor/117391/4
    """
    def __init__(self, data:Union[np.ndarray, coo_matrix, csr_matrix],
                 targets:Union[np.ndarray, coo_matrix, csr_matrix],
                 transform:bool = None):

        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data

        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets

        self.transform = transform

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]

    def get_all_data(self):
        return self.data, self.targets

def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = (coo.row, coo.col)
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s).to(DEVICE)

def sparse_batch_collate(batch):
    """
    Collate sparse batches into PyTorch tensors.
    """
    # batch[0] since it is returned as a one element list
    data_batch, targets_batch = batch[0]

    if type(data_batch[0]) == csr_matrix:
        data_batch = data_batch.tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch).to(DEVICE)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = targets_batch.tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.FloatTensor(targets_batch).to(DEVICE)
    return data_batch, targets_batch
