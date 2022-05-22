import random
import numpy as np
import torch

def same_seeds(seed: int = 18):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    """
    Early stops the training if validation loss/metrics doesn't improve after a given patience
    """
    def __init__(self, patience: int = 100, verbose: bool = True, delta: float = 0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 100
            verbose (bool): If True, prints a message for each validation loss improvement. Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        if loss == True:
            score = -metrics
        else:
            score = metrics
            
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose == True:
                    print(f"EarlyStopping") 
        else:
            self.best_score = score
            self.counter = 0
