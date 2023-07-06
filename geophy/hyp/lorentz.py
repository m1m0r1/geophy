import numpy as np
import torch


def origins(size: torch.Size, d: int, neg_k=1., dtype=None):
    inv_r = torch.sqrt(torch.as_tensor(neg_k, dtype=dtype))
    o = torch.cat([
        torch.ones(size + (1,), dtype=dtype) / inv_r,
        torch.zeros(size + (d,), dtype=dtype),
    ], dim=-1)
    return o

def origins_like(x: torch.Tensor, neg_k=1.):
    return origins(x.shape[:-1], d=x.shape[-1]-1, neg_k=neg_k, dtype=x.dtype)

def v_origins(size: torch.Size, d: int, dtype=None):
    vo = torch.zeros(size + (1 + d,), dtype=dtype)
    return vo

def fill_x0(x1: torch.Tensor, neg_k=1.):  # (..., d) -> (..., d+1)
    """ fill 0-th components of coordinates
    """
    r2 = 1. / torch.as_tensor(neg_k, dtype=x1.dtype)
    x0 = torch.sqrt(r2 + torch.square(x1).sum(dim=-1, keepdim=True))
    return torch.cat([x0, x1], dim=-1)

def fill_v0_o(v1: torch.Tensor):  # (..., d) -> (..., d+1)
    """ fill 0-th components of tangent vectors at origin
    """
    v0 = torch.zeros(v1.shape[:-1] + (1,), dtype=v1.dtype)
    return torch.cat([v0, v1], dim=-1)

def pseudo_inner(x: torch.Tensor, y: torch.Tensor, keepdim=False):  # (.., d+1) -> (..)
    i0 = [0] if keepdim else 0
    return - x[..., i0] * y[..., i0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=keepdim)

def pseudo_inner_o(x: torch.Tensor, neg_k=1., keepdim=False):  # (.., d+1) -> (..)
    r = 1./torch.sqrt(torch.as_tensor(neg_k, dtype=x.dtype))
    i0 = [0] if keepdim else 0
    return - r * x[..., i0]

def dist(x: torch.Tensor, y: torch.Tensor, neg_k=1., keepdim=False):
    inv_r2 = torch.as_tensor(neg_k, dtype=x.dtype)
    inv_r = torch.sqrt(inv_r2)
    return torch.acosh(torch.clamp(- inv_r2 * pseudo_inner(x, y, keepdim=keepdim), min=1.)) / inv_r

def dist_o(x: torch.Tensor, neg_k=1., keepdim=False):
    inv_r2 = torch.as_tensor(neg_k, dtype=x.dtype)
    inv_r = torch.sqrt(inv_r2)
    return torch.acosh(torch.clamp(- inv_r2 * pseudo_inner_o(x, neg_k=neg_k, keepdim=keepdim), min=1.)) / inv_r


def dist_mat(x: torch.Tensor, neg_k=1., keepdim=False):  # (n, d+1) -> (n, n)
    return dist(torch.unsqueeze(x, -2), torch.unsqueeze(x, -3), neg_k=neg_k, keepdim=keepdim)

def pseudo_norm_o(v: torch.Tensor, keepdim=False):
    vn = v[..., 1:].norm(dim=-1, keepdim=keepdim)
    return vn

def pseudo_norm(v: torch.Tensor, keepdim=False):
    vn2 = pseudo_inner(v, v, keepdim=keepdim)  # This should be positive
    vn2 = vn2.clamp(min=0.)   # added 23/1/26
    return torch.sqrt(vn2)

def expmap(x: torch.Tensor, v: torch.Tensor, neg_k=1.):
    vn = pseudo_norm(v, keepdim=True)
    inv_r =  torch.sqrt(torch.as_tensor(neg_k))
    vn_over_r = vn * inv_r
    vn_rate = sinh_div_ident(vn_over_r) #(torch.sinh(vn1) / vn1)
    y = torch.cosh(vn_over_r) * x + vn_rate * v
    return y

def expmap_o(v: torch.Tensor, neg_k=1.):
    vn = pseudo_norm_o(v, keepdim=True)
    inv_r =  torch.sqrt(torch.as_tensor(neg_k, dtype=v.dtype))
    vn_over_r = vn * inv_r
    o = origins_like(v, neg_k=neg_k)
    vn_rate = sinh_div_ident(vn_over_r)
    y = torch.cosh(vn_over_r) * o + vn_rate * v
    return y

def sinh_div_ident(x: torch.Tensor, eps=1e-6):
    val = torch.where(torch.abs(x) <= eps, 1. + x*x/6., torch.sinh(x) / x)
    return val

def acosh_div_sinh_acosh(v: torch.Tensor, eps=1e-8):
    ve = torch.sqrt((v*v - 1).clamp(0.)).clamp(eps)
    value = torch.log((ve + v)) / ve
    return torch.where(v >= 1. + eps, value, 1. - (v*v - 1.).clamp(0.)/6.)

def logmap(x: torch.Tensor, y: torch.Tensor, neg_k=1.):
    inv_r2 = torch.as_tensor(neg_k)
    p = pseudo_inner(x, y, keepdim=True)
    p_over_r2 = p * inv_r2
    v = acosh_div_sinh_acosh(- p_over_r2) * (y + p_over_r2 * x)  # TODO project here?
    return v

def logmap_o(x: torch.Tensor, neg_k=1.):
    inv_r2 = torch.as_tensor(neg_k)
    p = pseudo_inner_o(x, neg_k=neg_k, keepdim=True)
    p_over_r2 = p * inv_r2
    o = origins_like(x, neg_k=neg_k)
    v = acosh_div_sinh_acosh(- p_over_r2) * (x + p_over_r2 * o)  # v0 must be 0
    return v

def ptransp(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, neg_k=1.):
    inv_r2 = torch.as_tensor(neg_k, dtype=x.dtype)
    u = v + pseudo_inner(v, y, keepdim=True) / (1. - inv_r2 * pseudo_inner(x, y, keepdim=True)) * inv_r2 * (x + y)
    return u

def ptransp_o(x: torch.Tensor, v: torch.Tensor, neg_k=1.):
    o = origins_like(x, neg_k=neg_k)
    return ptransp(o, x, v, neg_k=neg_k)

def ptransp_to_o(x: torch.Tensor, v: torch.Tensor, neg_k=1.):
    inv_r = torch.sqrt(torch.as_tensor(neg_k, dtype=x.dtype))
    u1 = v[..., 1:] - v[..., 0:1]/(1. + inv_r * x[..., 0:1]) * inv_r * x[..., 1:]   # note the terms are not inv_r2 but inv_r
    return fill_v0_o(u1)

def ptransp_to_z_logmap(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    p = pseudo_inner(x, y, keepdim=True)
    v = y + p * x
    u = ptransp(x, z, v)
    return acosh_div_sinh_acosh(- p) * u

def ptransp_to_o_logmap(x: torch.Tensor, y: torch.Tensor, neg_k=1.):
    inv_r2 = torch.as_tensor(neg_k, dtype=x.dtype)
    p = pseudo_inner(x, y, keepdim=True)
    p_over_r2 = p * inv_r2
    v = y + p_over_r2 * x
    u = ptransp_to_o(x, v, neg_k=neg_k)
    return acosh_div_sinh_acosh(- p_over_r2) * u

def expmap_ptransp_o(x, v, neg_k=1.):
    inv_r = torch.sqrt(torch.as_tensor(neg_k, dtype=x.dtype))
    vn = pseudo_norm_o(v, keepdim=True)
    vn_over_r = vn * inv_r
    v_unit = v / vn_over_r.clamp(min=1e-8)
    y = torch.cosh(vn_over_r) * x + torch.sinh(vn_over_r) * ptransp_o(x, v_unit, neg_k=neg_k)
    return y
