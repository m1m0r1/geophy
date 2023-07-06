import logging
import torch
import torch.nn as nn
import numpy as np
from .hyp import lorentz


class WrappedNormal:
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, neg_k=1.):
        self.loc = loc  # (., dim + 1)
        self.scale = scale  # (., dim) 
        self.neg_k = torch.as_tensor(neg_k)  # negative curvature
        self._zero = torch.zeros(self.scale.shape, dtype=loc.dtype)  # (., dim)
        self._dim = loc.shape[-1] - 1   # Degrees of freedom in the manifold
        assert scale.shape[-1] == self._dim, (loc, scale)
        assert loc.shape[:-1] == scale.shape[:-1], (loc, scale)

    @torch.no_grad()
    def sample(self, shape):
        return self.rsample(shape=shape)

    def rsample(self, shape):
        v1 = torch.distributions.Normal(self._zero, self.scale).rsample(shape)   # shape + (., dim)
        v = lorentz.fill_v0_o(v1)
        x = self.loc[(None,) * len(shape) + (Ellipsis,)]  # for broadcasting  # shape + (., dim+1)
        y = lorentz.expmap_ptransp_o(x, v, neg_k=self.neg_k)
        return y

    def log_prob(self, x: torch.Tensor):
        v = lorentz.ptransp_to_o_logmap(self.loc, x, neg_k=self.neg_k)
        v1 = v[..., 1:]   # omit zero-th axis
        lls_norm = torch.distributions.Normal(self._zero, self.scale).log_prob(v1).sum(dim=-1)  # scale=(., dim), v1=(., dim) -> (.,)
        ps = lorentz.pseudo_inner(self.loc, x)   # (.,)
        inv_r2 = self.neg_k
        ps_over_r2 = ps * inv_r2
        wrap_terms = (self._dim - 1) * torch.log(lorentz.acosh_div_sinh_acosh(- ps_over_r2))  # (.,)
        lls = lls_norm + wrap_terms
        return lls  # (.,)


class WrappedMultivariateNormal:
    def __init__(self, loc: torch.Tensor, scale_tril: torch.Tensor, neg_k=1.):  # Sigma = LL^T
        self.loc = loc
        self.scale_tril = scale_tril
        self.neg_k = torch.as_tensor(neg_k)
        self._zero = torch.zeros(self.loc[..., 1:].shape, dtype=loc.dtype)  # (., dim)
        self._dim = loc.shape[-1] - 1
        assert self._dim == scale_tril.shape[-1] == scale_tril.shape[-2], (loc.shape, scale_tril.shape)

    @torch.no_grad()
    def sample(self, shape):
        return self.rsample(shape=shape)

    def rsample(self, shape):
        v1 = torch.distributions.MultivariateNormal(loc=self._zero, scale_tril=self.scale_tril).rsample(shape)   # shape + (., dim)
        v = lorentz.fill_v0_o(v1)
        x = self.loc[(None,) * len(shape) + (Ellipsis,)]  # for broadcasting  # shape + (., dim+1)
        y = lorentz.expmap_ptransp_o(x, v, neg_k=self.neg_k)
        return y

    def log_prob(self, x: torch.Tensor):   # (*extra, batch?, dim+1)
        extra_dim = len(x.shape) - len(self.loc.shape)
        assert extra_dim >= 0, (x.shape, self.loc.shape)
        extra_index = (None,) * extra_dim + (Ellipsis,)
        v = lorentz.ptransp_to_o_logmap(self.loc[extra_index], x, neg_k=self.neg_k)
        v1 = v[..., 1:]   # omit zero-th axis
        lls_norm = torch.distributions.MultivariateNormal(loc=self._zero, scale_tril=self.scale_tril).log_prob(v1)  # v1=(..., dim) -> (...,)
        ps = lorentz.pseudo_inner(self.loc[extra_index], x)   # (..., dim+1), (..., dim+1) (.,)
        ps_over_r2 = ps * self.neg_k
        wrap_terms = (self._dim - 1) * torch.log(lorentz.acosh_div_sinh_acosh(- ps_over_r2))  # (.,)
        lls = lls_norm + wrap_terms
        return lls  # (.,)
