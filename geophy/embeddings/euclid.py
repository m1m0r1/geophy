from .common import *
from .. import tree_dec

class EuclidTipEmbed(TipEmbedBase):
    """
    """
    def __init__(self, locs: torch.Tensor):
        self._locs = torch.as_tensor(locs)   # (n_tips, dim)

    def detach(self):
        return self.__class__(locs=self.locs.detach())

    @property
    def locs(self) -> torch.Tensor:
        return self._locs

    @property
    def vecs(self) -> torch.Tensor:
        return self._locs

    def get_distance_matrix(self) -> torch.Tensor:   # (n_tips, n_tips)
        tip_dist_mat = torch.cdist(self._locs, self._locs)
        return tip_dist_mat

    def get_utree_metric(self) -> TreeMetric:
        min_dist = 1e-7
        tip_dist_mat = self.get_distance_matrix()
        d = tip_dist_mat.shape[-1]
        try:
            tip_dist_mat = tip_dist_mat.cpu().detach().to(torch.float64).numpy()
            parents, blens = tree_dec.neighbor_joining(tip_dist_mat)  # returned as rooted
            blens = torch.as_tensor(blens, dtype=self._locs.dtype).clamp(min=min_dist)  # prevent possibly negative values of NJ
        except Exception as e:
            logging.error('tip_dist_mat: %s', tip_dist_mat)
            raise
        metric = TreeMetric(parent_indices=parents, branch_lengths=blens, rooted=True).get_unrooted()
        return metric

class NormalTipEmbedModel(TipEmbedModelBase):
    def __init__(self, n_tips,
            loc_init=None,  # shape (n_tips, dim)
            lscale_init=None,  # shape (n_tips, dim)
            embed_dim=2,
            dtype = None,
            ):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        self.n_tips = n_tips
        self.embed_dim = embed_dim
        if loc_init is None:
            loc_init = torch.zeros((n_tips, embed_dim), dtype=dtype)
        self.loc = nn.Parameter(loc_init)
        if lscale_init is None:
            lscale_init = torch.zeros((n_tips, embed_dim), dtype=dtype)  # (n_tips, dim)
        else:
            lscale_init = torch.from_numpy(np.broadcast_to(lscale_init, shape=(n_tips, embed_dim)).copy()).type(dtype)
        self.lscale = nn.Parameter(lscale_init)   # (n_tips, dim)

    def _get_dists(self):
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=self.loc, scale=torch.exp(self.lscale)), 1)

    def get_mean(self):
        return self.loc

    @property
    def mean_distance_matrix(self):
        dist_mat = torch.cdist(self.loc, self.loc)
        return dist_mat

    def rsample_embeds(self, mc_samples=1):
        try:
            dist = self._get_dists()
            locs = dist.rsample(torch.Size((mc_samples,)))   #  (mc_samples, n_tips, dim+1)
        except ValueError as e:
            raise
        return [EuclidTipEmbed(locs=locs[i]) for i in range(mc_samples)]

    def get_log_prob(self, embed: EuclidTipEmbed):
        locs = embed.locs   # (n_tips, d+1)
        dist = self._get_dists()
        ll = dist.log_prob(locs).sum()
        return ll

class NormalCondEmbedModel(CondEmbedModelBase):
    def __init__(self, n_tips,
            loc_init=None,  # shape (n_tips, dim + 1)
            lscale_init=None,  # shape (n_tips, dim)
            embed_dim=2,
            dtype = None,
            ):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        self.n_tips = n_tips
        self.embed_dim = embed_dim
        if loc_init is None:
            loc_init = torch.zeros((n_tips, embed_dim), dtype=dtype)
        self.loc = nn.Parameter(loc_init)
        if lscale_init is None:
            lscale_init = torch.zeros((n_tips, embed_dim), dtype=dtype)  # (n_tips, dim)
        else:
            lscale_init = torch.from_numpy(np.broadcast_to(lscale_init, shape=(n_tips, embed_dim)).copy()).type(dtype) #.clone().detach()
        self.lscale = nn.Parameter(lscale_init)   # (n_tips, dim)

    def get_mean(self):
        return self.loc

    def _get_dists(self, utree_metric: TreeMetricBase):
        return torch.distributions.Independent(
            torch.distributions.Normal(self.loc, torch.exp(self.lscale)), 1)

    def get_log_prob(self, embed: EuclidTipEmbed, utree_metric: TreeMetricBase):
        locs = embed.locs   # (n_tips, d+1)
        dist = self._get_dists(utree_metric)
        ll = dist.log_prob(locs).sum()
        return ll


class MultivariateNormalTipEmbedModel(TipEmbedModelBase):
    def __init__(self, n_tips,
            loc_init=None,  # shape (n_tips, dim)
            lscale_init=None,  # shape (n_tips, dim)
            embed_dim=2,
            dtype=None,
            ):
        super().__init__()
        self.n_tips = n_tips
        self.embed_dim = embed_dim
        dtype = dtype or torch.get_default_dtype()
        if loc_init is None:
            loc_init = torch.zeros((n_tips, embed_dim), dtype=dtype)
        self.loc = nn.Parameter(loc_init)
        if lscale_init is None:
            lscale_init = torch.zeros((n_tips, embed_dim), dtype=dtype)  # (n_tips, dim)
        else:
            lscale_init = torch.from_numpy(np.broadcast_to(lscale_init, shape=(n_tips, embed_dim)).copy()).type(dtype)
        self.lscale_diag = nn.Parameter(lscale_init)   # (n_tips, dim)
        non_diag_dim = embed_dim * (embed_dim - 1) // 2
        self.scale_non_diag = nn.Parameter(torch.zeros((n_tips, non_diag_dim), dtype=dtype)) # (n_tips, non_diag_dim)
        self._scale_non_diag_indices = tuple(torch.cat([
                torch.arange(self.n_tips)[:, None].repeat(1, non_diag_dim).flatten()[None, :],
                torch.tril_indices(embed_dim, embed_dim, -1).repeat(1, self.n_tips)
        ]))

    def get_mean(self):
        return self.loc

    def _get_dists(self):
        #scale_tril = torch.diag_embed(torch.exp(.5 * self.lscale))
        scale_tril = torch.diag_embed(torch.exp(self.lscale_diag))
        scale_tril[self._scale_non_diag_indices] = self.scale_non_diag.flatten()
        return torch.distributions.MultivariateNormal(loc=self.loc, scale_tril=scale_tril)

    @property
    def mean_distance_matrix(self):
        dist_mat = torch.cdist(self.loc, self.loc)
        return dist_mat

    def rsample_embeds(self, mc_samples=1):
        try:
            dist = self._get_dists()
            locs = dist.rsample(torch.Size((mc_samples,)))   #  (mc_samples, n_tips, dim+1)
        except ValueError as e:
            raise
        return [EuclidTipEmbed(locs=locs[i]) for i in range(mc_samples)]

    def get_log_prob(self, embed: EuclidTipEmbed):
        locs = embed.locs
        dist = self._get_dists()
        ll = dist.log_prob(locs).sum()
        return ll

class MultivariateNormalCondEmbedModel(CondEmbedModelBase):
    def __init__(self, n_tips,
            loc_init=None,  # shape (n_tips, dim)
            lscale_init=None,  # shape (n_tips, dim)
            embed_dim=2,
            dtype = None
            ):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        self.n_tips = n_tips
        self.embed_dim = embed_dim
        if loc_init is None:
            loc_init = torch.zeros((n_tips, embed_dim), dtype=dtype)
        self.loc = nn.Parameter(loc_init)       # adhoc solution (n_tips, dim)
        if lscale_init is None:
            lscale_init = torch.zeros((n_tips, embed_dim), dtype=dtype)  # (n_tips, dim)
        else:
            lscale_init = torch.from_numpy(np.broadcast_to(lscale_init, shape=(n_tips, embed_dim)).copy()).type(dtype)
        self.lscale_diag = nn.Parameter(lscale_init)   # (n_tips, dim)
        non_diag_dim = embed_dim * (embed_dim - 1) // 2
        self.scale_tril_non_diag = nn.Parameter(torch.zeros((n_tips, non_diag_dim), dtype=dtype)) # (n_tips, non_diag_dim)
        self._scale_tril_non_diag_indices = tuple(torch.cat([
                torch.arange(self.n_tips)[:, None].repeat(1, non_diag_dim).flatten()[None, :],
                torch.tril_indices(embed_dim, embed_dim, -1).repeat(1, self.n_tips)
        ]))

    def get_mean(self):
        return self.loc

    def _get_dists(self):
        scale_tril = torch.diag_embed(torch.exp(self.lscale_diag))
        scale_tril[self._scale_tril_non_diag_indices] = self.scale_tril_non_diag.flatten()
        return torch.distributions.MultivariateNormal(loc=self.loc, scale_tril=scale_tril)

    def get_log_prob(self, embed: EuclidTipEmbed, utree_metric: TreeMetricBase):
        locs = embed.locs   # (n_tips, d+1)
        dist = self._get_dists()
        ll = dist.log_prob(locs).sum()
        return ll
