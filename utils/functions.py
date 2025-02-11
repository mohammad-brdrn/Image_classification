"""
This Script applies a fix seed on random generators,
ensuring the reproducibility of the results.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 1234) -> None:
    """
    Applies a fixed seed for reproducibility.
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
