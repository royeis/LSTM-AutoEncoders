import torch
from torch.utils.data.dataset import TensorDataset


def prepare_tensor_dataset(dataset):
    return TensorDataset(torch.Tensor(dataset))
