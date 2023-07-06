from .common import *
from .. import wrapped_normal
from ..hyp import lorentz
from .. import tree_dec

class LorentzTipEmbed(TipEmbedBase):
    """
    """
    def __init__(self, locs: torch.Tensor, neg_k=1.):
        #import pdb; pdb.set_trace()
        self._locs = torch.as_tensor(locs)   # (n_tips, dim+1)
        self._vecs = lorentz.logmap_o(self._locs)[..., 1:]  # Euclidean representation
        self._neg_k = torch.as_tensor(neg_k)

    def detach(self):
        return self.__class__(locs=self.locs.detach(), neg_k=self.neg_k)

    @property
    def neg_k(self) -> torch.Tensor:
        return self._neg_k

    @property
    def locs(self) -> torch.Tensor:
        return self._locs

    @property
    def vecs(self) -> torch.Tensor:
        return self._vecs

    def get_distance_matrix(self) -> torch.Tensor:   # (n_tips, n_tips)
        tip_coords = self._locs
        tip_dist_mat = lorentz.dist(tip_coords[:, None, :], tip_coords[None, :, :], neg_k=self.neg_k, keepdim=False)
        return tip_dist_mat

    def get_utree_metric(self) -> TreeMetric:
        min_dist = 1e-7
        tip_dist_mat = self.get_distance_matrix()
        try:
            tip_dist_mat = tip_dist_mat.cpu().detach().to(torch.float64).numpy()
            parents, blens = tree_dec.neighbor_joining(tip_dist_mat)  # returned as rooted
            blens = torch.as_tensor(blens, dtype=self._locs.dtype).clamp(min=min_dist)  # prevent possibly negative values of NJ
        except Exception as e:
            logging.error('tip_dist_mat: %s', tip_dist_mat)
            raise
        metric = TreeMetric(parent_indices=parents, branch_lengths=blens, rooted=True).get_unrooted()
        return metric

class WrappedNormalTipEmbedModel(TipEmbedModelBase):
    def __init__(self, n_tips,
            loc_init=None,  # lorentz coordinate with shape (n_tips, dim + 1)
            lscale_init=None,  # tangent vector at the origin with shape (n_tips, dim)
            embed_dim=2,
            neg_k=1.,
            dtype=None,
            ):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        self.n_tips = n_tips
        self.embed_dim = embed_dim
        self.neg_k = torch.as_tensor(neg_k, dtype=dtype)
        if loc_init is None:
            loc_init = lorentz.origins((n_tips,), d=embed_dim)
        self.loc1 = nn.Parameter(loc_init[:, 1:])
        if lscale_init is None:
            lscale_init = torch.zeros((n_tips, embed_dim), dtype=dtype)  # (n_tips, dim)
        else:
            lscale_init = torch.from_numpy(np.broadcast_to(lscale_init, shape=(n_tips, embed_dim)).copy()).type(dtype) #.clone().detach()
        self.lscale = nn.Parameter(lscale_init)   # (n_tips, dim)

    def _get_dists(self):
        loc = lorentz.fill_x0(self.loc1, neg_k=self.neg_k)
        return wrapped_normal.WrappedNormal(loc=loc, scale=torch.exp(self.lscale), neg_k=self.neg_k)

    def get_mean(self):
        loc = lorentz.fill_x0(self.loc1, neg_k=self.neg_k)
        return loc

    @property
    def mean_distance_matrix(self):
        loc = lorentz.fill_x0(self.loc1, neg_k=self.neg_k)
        dist_mat = lorentz.dist_mat(loc, neg_k=self.neg_k)
        return dist_mat

    def rsample_embeds(self, mc_samples=1):
        try:
            dist = self._get_dists()
            locs = dist.rsample(torch.Size((mc_samples,)))   #  (mc_samples, n_tips, dim+1)
        except ValueError as e:
            raise
        return [LorentzTipEmbed(locs=locs[i]) for i in range(mc_samples)]   # TODO return embed list structure

    def get_log_prob(self, embed: LorentzTipEmbed):  # TODO take batched embeds
        locs = embed.locs   # (n_tips, d+1)
        dist = self._get_dists()
        ll = dist.log_prob(locs).sum()
        return ll



class WrappedNormalCondEmbedModel(CondEmbedModelBase):
    def __init__(self, n_tips,
            loc_init=None,  # lorentz coordinate with shape (n_tips, dim + 1)
            lscale_init=None,  # tangent vector at the origin with shape (n_tips, dim)
            embed_dim=2,
            neg_k=1.,
            dtype=None,
            ):
        super().__init__()
        dtype = torch.get_default_dtype()
        self.n_tips = n_tips
        self.embed_dim = embed_dim
        self.neg_k = torch.as_tensor(neg_k, dtype=dtype)
        if loc_init is None:
            loc_init = lorentz.origins((n_tips,), d=embed_dim, neg_k=neg_k, dtype=dtype)    # (dim + 1,)
        self.loc1 = nn.Parameter(loc_init[:, 1:])       # adhoc solution (n_tips, dim)
        if lscale_init is None:
            lscale_init = torch.zeros((n_tips, embed_dim))  # (n_tips, dim)
        else:
            lscale_init = torch.from_numpy(np.broadcast_to(lscale_init, shape=(n_tips, embed_dim)).copy()).type(dtype) #.clone().detach()
        self.lscale = nn.Parameter(lscale_init)   # (n_tips, dim)

    def get_mean(self):
        loc = lorentz.fill_x0(self.loc1, neg_k=self.neg_k)
        return loc

    def _get_dists(self, utree_metric: TreeMetricBase):
        loc = lorentz.fill_x0(self.loc1, neg_k=self.neg_k)
        return wrapped_normal.WrappedNormal(loc, torch.exp(self.lscale), neg_k=self.neg_k)
    def get_log_prob(self, embed: LorentzTipEmbed, utree_metric: TreeMetricBase):
        locs = embed.locs   # (n_tips, d+1)
        dist = self._get_dists(utree_metric)
        ll = dist.log_prob(locs).sum()
        return ll


class WrappedMultivariateNormalTipEmbedModel(TipEmbedModelBase):
    def __init__(self, n_tips,
            loc_init=None,  # lorentz coordinate with shape (n_tips, dim + 1)
            lscale_init=None,  # tangent vector at the origin with shape (n_tips, dim)
            embed_dim=2,
            neg_k=1.,
            dtype=None,
            ):
        super().__init__()
        dtype = torch.get_default_dtype()
        self.n_tips = n_tips
        self.embed_dim = embed_dim
        self.neg_k = torch.as_tensor(neg_k)
        if loc_init is None:
            loc_init = lorentz.origins((n_tips,), d=embed_dim, neg_k=neg_k, dtype=dtype)    # (dim + 1,)
        self.loc1 = nn.Parameter(loc_init[:, 1:])
        if lscale_init is None:
            lscale_init = torch.zeros((n_tips, embed_dim), dtype=dtype)  # (n_tips, dim)
        else:
            lscale_init = torch.from_numpy(np.broadcast_to(lscale_init, shape=(n_tips, embed_dim)).copy()).type(dtype) #.clone().detach()
        self.lscale_diag = nn.Parameter(lscale_init)   # (n_tips, dim)
        non_diag_dim = embed_dim * (embed_dim - 1) // 2
        self.scale_non_diag = nn.Parameter(torch.zeros((n_tips, non_diag_dim))) # (n_tips, non_diag_dim)
        self._scale_non_diag_indices = tuple(torch.cat([
                torch.arange(self.n_tips)[:, None].repeat(1, non_diag_dim).flatten()[None, :],
                torch.tril_indices(embed_dim, embed_dim, -1).repeat(1, self.n_tips)
        ]))

    def _get_dists(self):
        loc = lorentz.fill_x0(self.loc1, neg_k=self.neg_k)
        scale_tril = torch.diag_embed(torch.exp(self.lscale_diag))
        scale_tril[self._scale_non_diag_indices] = self.scale_non_diag.flatten()
        return wrapped_normal.WrappedMultivariateNormal(loc=loc, scale_tril=scale_tril, neg_k=self.neg_k)

    def get_mean(self):
        loc = lorentz.fill_x0(self.loc1, neg_k=self.neg_k)
        return loc

    @property
    def mean_distance_matrix(self):
        loc = lorentz.fill_x0(self.loc1, neg_k=self.neg_k)
        dist_mat = lorentz.dist_mat(loc, neg_k=self.neg_k)
        return dist_mat

    def rsample_embeds(self, mc_samples=1):
        try:
            dist = self._get_dists()
            locs = dist.rsample(torch.Size((mc_samples,)))   #  (mc_samples, n_tips, dim+1)
        except ValueError as e:
            raise
        return [LorentzTipEmbed(locs=locs[i]) for i in range(mc_samples)]

    def get_log_prob(self, embed: LorentzTipEmbed):
        locs = embed.locs   # (n_tips, d+1)
        dist = self._get_dists()
        ll = dist.log_prob(locs).sum()
        return ll

class WrappedMultivariateNormalCondEmbedModel(CondEmbedModelBase):
    def __init__(self, n_tips,
            loc_init=None,  # lorentz coordinate with shape (n_tips, dim + 1)
            lscale_init=None,  # tangent vector at the origin with shape (n_tips, dim)
            embed_dim=2,
            neg_k=1.,
            dtype=None,
            ):
        super().__init__()
        dtype = torch.get_default_dtype()
        self.n_tips = n_tips
        self.embed_dim = embed_dim
        self.neg_k = torch.as_tensor(neg_k, dtype=dtype)
        if loc_init is None:
            loc_init = lorentz.origins((n_tips,), d=embed_dim, dtype=dtype)    # (dim + 1,)
        self.loc1 = nn.Parameter(loc_init[:, 1:])       # adhoc solution (n_tips, dim)
        if lscale_init is None:
            lscale_init = torch.zeros((n_tips, embed_dim), dtype=dtype)  # (n_tips, dim)
        else:
            lscale_init = torch.from_numpy(np.broadcast_to(lscale_init, shape=(n_tips, embed_dim)).copy()).type(dtype) #.clone().detach()
        self.lscale_diag = nn.Parameter(lscale_init)   # (n_tips, dim) 
        non_diag_dim = embed_dim * (embed_dim - 1) // 2
        self.scale_tril_non_diag = nn.Parameter(torch.zeros((n_tips, non_diag_dim), dtype=dtype)) # (n_tips, non_diag_dim)
        self._scale_tril_non_diag_indices = tuple(torch.cat([
                torch.arange(self.n_tips)[:, None].repeat(1, non_diag_dim).flatten()[None, :],
                torch.tril_indices(embed_dim, embed_dim, -1).repeat(1, self.n_tips)
        ]))

    def get_mean(self):
        loc = lorentz.fill_x0(self.loc1, neg_k=self.neg_k)
        return loc

    def _get_dists(self):
        loc = lorentz.fill_x0(self.loc1, neg_k=self.neg_k)
        scale_tril = torch.diag_embed(torch.exp(self.lscale_diag))
        scale_tril[self._scale_tril_non_diag_indices] = self.scale_tril_non_diag.flatten()
        return wrapped_normal.WrappedMultivariateNormal(loc=loc, scale_tril=scale_tril, neg_k=self.neg_k)

    def get_log_prob(self, embed: LorentzTipEmbed, utree_metric: TreeMetricBase):
        locs = embed.locs   # (n_tips, d+1)
        dist = self._get_dists()
        ll = dist.log_prob(locs).sum()
        return ll
