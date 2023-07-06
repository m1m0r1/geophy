import numpy as np
import torch
from typing import *
from .nj import nj

def peel2parents(peel) -> List:
    # (left, right, parent)
    parent_indices = [-1] * int(peel.max())
    for lrp in peel:
        parent_indices[lrp[0]] = parent_indices[lrp[1]] = lrp[2]
    return parent_indices

def neighbor_joining(dist_mat: np.ndarray) -> Tuple[List, List]:
    peel, blens = nj(dist_mat)
    # A list of internal nodes [(left_idx, right_idx, left_len, right_len)]
    parents = list(peel2parents(peel))
    return parents, blens  # return unrooted node with pseudo root node
