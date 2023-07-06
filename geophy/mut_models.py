import numpy as np
import torch
# From https://github.com/zcrabbit/vbpi-torch/blob/ff86cf0c47a5753f5cc5b4dfe0b6ed783ab22669/unrooted/rateMatrix.py#L5-L20
# From https://github.com/zcrabbit/vbpi-torch/blob/ff86cf0c47a5753f5cc5b4dfe0b6ed783ab22669/unrooted/phyloModel.py#L51-L52

def decompJC(symm=False):
    # pA = pG = pC = pT = .25
    pden = np.array([.25, .25, .25, .25])
    rate_matrix_JC = 1.0/3 * np.ones((4,4))
    for i in range(4):
        rate_matrix_JC[i,i] = -1.0

    if not symm:
        D_JC, U_JC = np.linalg.eig(rate_matrix_JC)
        U_JC_inv = np.linalg.inv(U_JC)
    else:
        D_JC, W_JC = np.linalg.eigh(np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_JC), np.diag(np.sqrt(1.0/pden))))
        U_JC = np.dot(np.diag(np.sqrt(1.0/pden)), W_JC)
        U_JC_inv = np.dot(W_JC.T, np.diag(np.sqrt(pden)))

    return D_JC, U_JC, U_JC_inv, rate_matrix_JC

class MutationModel:
    def get_trans_mats(self, blens):   # (n_branches,) -> (n_branches, n_from_chars, n_to_chars)
        raise NotImplementedError

def _get_trans_mats(blens, D, U, U_inv):
    blen_D = blens[:, None] * D[None, :]   # (n_branches, n_chars)
    trans_mats = torch.einsum('ij, bj, jk -> bik', U, torch.exp(blen_D), U_inv).clamp(min=0.)   # (n_branches, n_chars, n_chars)
    return trans_mats


class JCMutModel(MutationModel):
    def __init__(self, dtype=None):
        dtype = dtype or torch.get_default_dtype()
        self.base_prob = torch.tensor([.25] * 4).to(dtype)
        self.D, self.U, self.U_inv, self.rateM = decompJC(symm=True)
        self.D = torch.from_numpy(self.D).to(dtype)
        self.U = torch.from_numpy(self.U).to(dtype)
        self.U_inv = torch.from_numpy(self.U_inv).to(dtype)

    def get_trans_mats(self, blens):
        return _get_trans_mats(blens, self.D, self.U, self.U_inv)
