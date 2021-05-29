import torch
from torch.utils.data.dataset import TensorDataset
import random
import numpy as np

def prepare_tensor_dataset(dataset):
    return TensorDataset(torch.Tensor(dataset))

def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)