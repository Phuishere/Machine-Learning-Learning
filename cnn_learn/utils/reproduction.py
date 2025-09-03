import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)                          # Python built-in random
    np.random.seed(seed)                       # NumPy
    torch.manual_seed(seed)                    # PyTorch CPU
    torch.cuda.manual_seed(seed)               # Current GPU
    torch.cuda.manual_seed_all(seed)           # All GPUs
    torch.backends.cudnn.deterministic = True  # Make cuDNN deterministic
    torch.backends.cudnn.benchmark = False     # Disable performance optimization that can introduce randomness