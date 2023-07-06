from typing import *
import logging
import numpy as np
import torch
import torch.nn as nn
from .gnn_models import GNNStack, IDConv, MeanStdPooling
# From https://github.com/zcrabbit/vbpi-gnn/blob/main/gnn_branchModel.py
# Modified

class BranchModelBase(nn.Module):
    """
    """
    @property
    def state_size(self) -> torch.Size:
        raise NotImplementedError

    @property
    def parent_indices(self):   # in post order
        raise NotImplementedError

    def get_rparam_sizes(self):
        """ Returns dict whose keys are param names and values are the size of the differentiable variables
        """
        raise NotImplementedError

    def sample_branch_lengths(self):
        raise NotImplementedError


class LogNormalBranchModel(BranchModelBase):          
    """
    blen = scale * org_blen
    """
    def __init__(self, n_tips, hidden_dim=100, num_layers=1, gnn_type='gcn', aggr='sum', project=False, bias=True,
            lscale=0., lsigma=0.,
            **kwargs):
        super().__init__()
        self.n_tips = n_tips
        self.n_nodes = 2 * n_tips - 2
        self.n_branches = 2 * n_tips - 3    # the number of branches of an unrooted binary tree
            #logging.error('log_blens: %s', log_blens)

        self.lscale_mu = nn.Parameter(torch.as_tensor(lscale))
        self.lsigma = nn.Parameter(torch.as_tensor(lsigma))
        logging.debug('lscale_mu, lsigma: %s, %s', self.lscale_mu, self.lsigma)  # ok
        if torch.any(torch.isnan(self.lscale_mu)):
            logging.error('found nan in lscale_mu: %s', locals())
            raise RuntimeError
        if torch.any(torch.isnan(self.lsigma)):
            logging.error('found nan in lsigma: %s', locals())
            raise RuntimeError
        if gnn_type == 'identity':
            self.gnn = IDConv()
        elif gnn_type != 'ggnn':
            self.gnn = GNNStack(self.n_tips, hidden_dim, num_layers=num_layers, bias=bias, gnn_type=gnn_type, aggr=aggr, project=project)
        else:
            raise NotImplementedError

        if gnn_type == 'identity':
            self.mean_std_net = MeanStdPooling(self.n_tips, hidden_dim, bias=bias)
        else:
            self.mean_std_net = MeanStdPooling(hidden_dim, hidden_dim, bias=bias)

        self.mean_std_net.init_last_layer() #readout.apply(init_weights)   # set zero as default values

    @property
    def state_size(self) -> torch.Size:
        return torch.Size((2,))

    @staticmethod
    def params_for_distance_fit(target_dist_mat: torch.Tensor, loc_dist_mat: torch.Tensor):
        N = target_dist_mat.shape[0]
        assert target_dist_mat.shape == (N, N)
        assert loc_dist_mat.shape == (N, N)
        idxs = torch.tril_indices(N, N, -1)
        non_diag_sel = idxs[0], idxs[1]
        logging.info('Range of target_dist_mat: (%s, %s)', target_dist_mat[non_diag_sel].min(), target_dist_mat[non_diag_sel].max())
        logging.info('Range of loc_dist_mat: (%s, %s)', loc_dist_mat[non_diag_sel].min(), loc_dist_mat[non_diag_sel].max())
        dist_eps = 1e-7   # critical for DS6-8
        unit = torch.eye(N)   # dummy for avoiding log errors of the diagonal elements (TODO is dist_eps enough?)
        ldiff = torch.log(torch.clamp(unit + target_dist_mat, min=dist_eps)) - torch.log(torch.clamp(unit + loc_dist_mat, min=dist_eps))
        lscale = ldiff.sum() / (N * (N - 1))
        sigma2 = torch.square(ldiff - lscale).sum() / (N * (N - 1))
        lsigma = .5 * torch.log(sigma2)
        assert not torch.any(torch.isnan(lscale)), locals()
        assert not torch.any(torch.isnan(lsigma)), locals()
        return lscale, lsigma

    def predict_params(self, node_features: torch.Tensor, edge_indexes: torch.Tensor):
        node_features = self.gnn(node_features, edge_indexes)
        #logging.debug('predict_params %s', node_features)   # ok
        mu, lsigma = self.mean_std_net(node_features, edge_indexes[:, :-1, 0])   # this sets parent indexes
        #logging.debug('predict_params %s, %s', mu, lsigma)  # ok
        #logging.debug('lscale_mu, lsigma: %s, %s', self.lscale_mu, self.lsigma)  # ok
        mu = mu + self.lscale_mu
        lsigma = lsigma + self.lsigma
        #logging.debug('predict_params %s, %s', mu, lsigma)  # bad
        return mu, lsigma

    #TODO efficient implementation with multiple trees
    def sample_branch_lengths(self, node_features: torch.Tensor, edge_indexes: torch.Tensor):  #List[embeddings.TreeTopologyBase)]:
        mc_samples = node_features.shape[0]
        assert node_features.shape[1] == self.n_nodes
        assert node_features.shape[2] == self.n_tips
        assert edge_indexes.shape[1] == self.n_nodes
        assert edge_indexes.shape[2] == 3
        #with torch.no_grad():   # Ignoring gradient back prop to embedding samples here (This is required to ignore graident back to embedding through branch lengths)
        rvs = torch.normal(
                mean=torch.zeros((mc_samples, self.n_branches,)),
                std=torch.ones((mc_samples, self.n_branches,)))

#       node_features, edge_index = self.node_embedding(tree)
#       node_features = self.gnn(node_features, edge_index)
#
#       return self.mean_std_net(node_features, edge_index[:-1, 0])
        mu, lsigma = self.predict_params(node_features, edge_indexes)
        assert mu.shape == lsigma.shape, (mu.shape, lsigma.shape)
        assert mu.shape == (mc_samples, self.n_branches), mu.shape
        #sigma = torch.exp(self.lsigma + lsigma)
        sigma = torch.exp(lsigma)
        log_blens = mu + sigma * rvs
        blens = torch.exp(log_blens)
        if torch.any(torch.isnan(blens)):
            logging.error('blens: %s', blens)
            #logging.error('log_blens: %s', log_blens)
            logging.error('mu: %s', mu)
            logging.error('sigma: %s', sigma)
            logging.error('lsigma: %s', lsigma)
            logging.error('node_features: %s', node_features)
            #logging.error('edge_indexes: %s', edge_indexes)
            logging.error('rvs: %s', rvs)
            raise ValueError(blens)
        # density of LogNormal
        #logqs = - (log_blens + self.lsigma + .5 * np.log(2 * np.pi))
        #sigma2 = torch.square(sigma) #exp(self.lsigma * 2)
        #logqs -= .5 * (log_blens - mu)**2 / sigma2  # simplified thanks to repara trick
        #logq = logqs.sum()
        log_qs = - (log_blens + lsigma + .5 * np.log(2 * np.pi))
        log_qs -= .5 * (rvs**2)  # simplified thanks to repara trick
        log_q = log_qs.sum(dim=-1)
        #print (blens.min(), logq)
        #new_metric = tree_metric.clone(branch_lengths=blens)
        return blens, {
                'log_q': log_q, 
                'state': torch.as_tensor([self.lscale_mu, self.lsigma]), #rvs
                }   # TODO also set global scale parameter?


