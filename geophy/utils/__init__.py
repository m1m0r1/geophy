import random
import os
import torch
import torch.nn
import numpy as np
import pandas as pd
from ..mut_models import MutationModel
from ..hyp import lorentz
import logging as logger
from omegaconf import OmegaConf


def parse_config(config_path, overrides=None):
    logger.info('Parse config from {}'.format(config_path))
    cfg = OmegaConf.load(config_path)
    #logger.info(f'Original config:\n{OmegaConf.to_yaml(cfg)}')
    if overrides is not None:
        logger.info(f'Overriding with {overrides}')
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        #logger.info(f'Updated Config:\n{OmegaConf.to_yaml(cfg)}')
    return cfg


def load_seqs(input_path):
    # Returns names, seqs
    if input_path.endswith('.txt'):
        seq_data = pd.read_csv(input_path, sep='\t')
        return seq_data['name'], seq_data['seq']
    if input_path.endswith('.nex'):
        import dendropy
        ds = dendropy.DataSet.get(path=input_path, schema='nexus')
        names = [taxon.label for taxon, seq in ds.char_matrices[0].items()]
        seqs = [str(seq) for taxon, seq in ds.char_matrices[0].items()]
        return names, seqs
    raise NotImplementedError

# cannot fix seed for np.random.default_rng()
# See https://github.com/qhd1996/seed-everything
def init_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


_dtype_map = {
    'float32': torch.float32,
    'float64': torch.float64,
}

def set_default_dtype(dtype):
    if isinstance(dtype, str):
        dtype = _dtype_map[dtype]
    logger.info('Set default dtype to {}'.format(dtype))
    torch.set_default_dtype(dtype)

class with_default_dtype:
    def __init__(self, dtype):
        self._dtype = dtype

    def __enter__(self):
        self._org_dtype = torch.get_default_dtype()
        set_default_dtype(self._dtype)

    def __exit__(self, *args, **kws):
        set_default_dtype(self._org_dtype)

from scipy.special import gammaln
def log_factorial2(n):
    k = n // 2   # n = 2k => k, n = 2k+1 => k  (n=1 => k=0, n=3 => k=1)
    if n % 2 == 0:
        return k * np.log(2) + gammaln(k+1)
    if n % 2 == 1:
        return gammaln(n+1) - k * np.log(2) - gammaln(k+1)
    raise NotImplementedError

def get_hamming_dist_mat(seq_tensor: torch.Tensor, normalize=True):   # -> dist_mat: (N, N)
    """
    Count mismatch of all the pairs at each position (N, N, M), then sum up those for each pair

    - seq_tensor: torch.tensor(N, M, n_chars)

    # TODO reduce memory requirement for large N and M
    """
    M = seq_tensor.shape[1]
    seq_pair_pos_dot = torch.einsum('imc, jmc -> ijm', seq_tensor, seq_tensor)
    seq_pair_mismatch = (seq_pair_pos_dot < .5).sum(dim=-1)   # (N, N)
    if normalize:
        seq_pair_mismatch = seq_pair_mismatch / M
    return seq_pair_mismatch


# Euclidean coordinate # TODO check
def get_mds(dist_mat: torch.Tensor, rank=2):  # (N, N) -> (N, rank), data  # Euclidean model
    D2 = dist_mat * dist_mat
    N = dist_mat.shape[0]
    C = (torch.eye(N, N) - torch.ones((N, 1)) * torch.ones((1, N)) / N).type(dist_mat.dtype)
    K = - .5 * C.mm(D2).mm(C)
    L, Q = torch.linalg.eigh(- K)   # ascending order (thus, negatives to positives) for Y, thus, descending order for -K
    x = Q[:, :rank] * (- L[:rank]).clamp(min=0) ** .5 # x1 = Q[:, :rank] 
    return x, {'L': L, 'Q': Q}

# Hyperbolid coordinate
def get_hmds(dist_mat: torch.Tensor, rank=2):  # (N, N) -> (N, rank+1), data  # hyperboloid model
    Y = torch.cosh(dist_mat)
    L, Q = torch.linalg.eigh(Y)   # ascending order (thus, negatives to positives) for Y, thus, descending order for -Y
    x1 = Q[:, :rank] * ((- L[:rank]).clamp(min=0) ** .5) # x1 = Q[:, :rank] 
    x = lorentz.fill_x0(x1)   # (N, rank+1)
    #x0 = torch.sqrt(1 + x1.norm(dim=-1, keepdim=True)**2)  # (N, 1)
    #x = torch.hstack([x0, x1])
    return x, {'L': L, 'Q': Q}


def sum_product_ll(seq_tensor: torch.Tensor, mut_model: MutationModel, parent_indices, branch_lengths):
    """
    - seq_tensor: torch.tensor(N, M, n_chars)
    - mut_model.get_trans_mats(blens) : n_chars -> n_chars

    -> log_likelihood (tensor)
    """
    if seq_tensor.shape[1] == 0:  # empty sequence
        return torch.tensor(0.)
    n_tips = seq_tensor.shape[0]
    parents = parent_indices   # 2N-3 or 2N-2 (unrooted or rooted). Indices are children.
    assert not isinstance(parents, torch.Tensor), parents
    root_idx = len(parents) or None #- 1
    try:
        assert n_tips >= 2, 'Minimum #tips is 2'
        assert min(parents) == n_tips, 'The first parent index should be the next index of the last tip index'
        assert max(parents) == root_idx, 'The root index should be the maximum index of parents'
        assert len(branch_lengths) == len(parents), f'The number of branches should be the number of parent indices {len(branch_lengths)} != {len(parents)}'
        if len(parents) == 2 * n_tips - 2:  # rooted
            assert n_tips >= 2
        elif len(parents) == 2 * n_tips - 3: # unrooted
            assert n_tips >= 3
        else:
            logger.error(f'Unexpected input; n_tips: {n_tips}, parents: {parents}')
            raise NotImplementedError
    except Exception as e:
        logger.error('locals: %s', locals())
        logger.error(e)
        raise

    blens = branch_lengths # n_branch = 2N-3 or 2N-2 (unrooted or rooted). Indices are children.
    base_prob = mut_model.base_prob  # (n_chars,)
    trans_mats = mut_model.get_trans_mats(blens)   # (n_branch, n_pa_chars, n_ch_chars)
    #import pdb; pdb.set_trace()
    #assert root_idx > 0, 'Number of parent nodes should be greater than 0'

    msgs = {}
    for n, pa_n in enumerate(parents):
        msg_ch = seq_tensor[n] if n < n_tips else msgs[n]   # (seq_len, n_chars)
        msg_pa = torch.mm(msg_ch, trans_mats[n].T)   # (seq_len, n_ch_chars), (n_pa_chars, n_ch_chars)
        if pa_n in msgs:
            msgs[pa_n] = msgs[pa_n] * msg_pa
        else:
            msgs[pa_n] = msg_pa

    msg = msgs[root_idx]  # (seq_len, n_chars)
    ll = torch.log(torch.mv(msg, base_prob)).sum()
    return ll


def get_grad_values(model):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def torch_safe_save(state, path):
    torch.save(state, path + '.tmp')
    os.rename(path + '.tmp', path)


def get_cached_file(cache, path, reader):
    mtime = os.path.getmtime(path)
    if path in cache:
        if cache[path]['mtime'] == mtime:
            logger.info('Latest cache found for {}'.format(path))
            return cache[path]['data']
    cache[path] = {
        'mtime': mtime,
        'data': reader(path),
    }
    return cache[path]['data']
