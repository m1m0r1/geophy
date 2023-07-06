from .. import embeddings
from .. import utils
from .lax import LAXModel
from ..branch_models import BranchModelBase
from ..mut_models import MutationModel
from ..seq_data import SequenceData

import numpy as np
import torch
import torch.nn as nn
import logging as logger

def get_loo_variable(x, dim=0):
    n = x.shape[dim]
    assert n > 1
    return (n * x - x.sum(dim=dim, keepdim=True)) / (n-1)


class GeoPhyModel(nn.Module):
    def __init__(self, seq_data: SequenceData, mut_model: MutationModel, tip_embed_model: embeddings.TipEmbedModelBase, branch_model: BranchModelBase, cond_embed_model: embeddings.CondEmbedModelBase): 
        super().__init__()
        self._mut_model = mut_model
        self.seq_len = seq_data.seq_len
        self.n_chars = seq_data.n_chars
        self.n_tips = len(seq_data)
        self.utree_log_prior_value = - utils.log_factorial2(2 * self.n_tips - 5)
        self.tip_embed_model = tip_embed_model
        self.branch_model = branch_model
        self.branch_prior = torch.distributions.Exponential(rate=10.)  # 1/scale
        self.cond_embed_model = cond_embed_model
        self._seq_tensor = torch.as_tensor(seq_data.get_tensor())

    def sample_states(self, mc_samples=10, inv_temp=1.):
        tip_embeds = self.tip_embed_model.rsample_embeds(mc_samples)
        tip_embed_log_qs = torch.stack([self.tip_embed_model.get_log_prob(tip_embed) for tip_embed in tip_embeds])  # (mc_samples,)
        tip_embeds_detach = [e.detach() for e in tip_embeds]
        tip_embed_detach_log_qs = torch.stack([self.tip_embed_model.get_log_prob(te) for te in tip_embeds_detach])  # (mc_samples,)
        # Ignoring gradient backprop for embedded samples here
        with torch.no_grad():
            utree_samples = [te.get_utree_metric() for te in tip_embeds_detach]
            utree_log_ps = torch.as_tensor([self.utree_log_prior_value for utree in utree_samples])  # default choice

        node_features = torch.stack([embeddings.get_learnable_feature(utree) for utree in utree_samples])   # (mc_samples, n_nodes, n_features)
        if torch.any(torch.isnan(node_features)):
            logger.error('found %s nan in node_features: %s', torch.isnan(node_features).sum(), node_features)
            raise RuntimeError()
        edge_indexes = torch.stack([embeddings.get_edge_index_tensor(utree) for utree in utree_samples])    # (mc_samples, n_nodes, 2)
        branch_lens, branch_d = self.branch_model.sample_branch_lengths(node_features=node_features, edge_indexes=edge_indexes)
        n_nodes = self.n_tips * 2 - 2
        assert node_features.shape == (mc_samples, n_nodes, self.n_tips), node_features.shape
        assert edge_indexes.shape == (mc_samples, n_nodes, 3), edge_indexes.shape
        assert branch_lens.shape == (mc_samples, n_nodes-1), branch_lens.shape
        assert branch_d['log_q'].shape == (mc_samples,), branch_d['log_q']

        branch_log_qs = branch_d['log_q']  #torch.stack([d['log_q'] for _, d in branch_samples])  # (mc_samples,)
        branch_log_ps = self.branch_prior.log_prob(branch_lens).sum(dim=-1)  # (mc_samples,) #)
        tip_embed_detach_log_rs = torch.stack([
            self.cond_embed_model.get_log_prob(embed=te, utree_metric=utree)
            for te, utree in zip(tip_embeds_detach, utree_samples)])

        lls = torch.stack([
            utils.sum_product_ll(
                seq_tensor=self._seq_tensor, mut_model=self._mut_model,
                parent_indices=utree.parent_indices,
                branch_lengths=blens)
                    for utree, blens in zip(utree_samples, branch_lens)])  # (mc_samples,)

        assert lls.shape == (mc_samples,)
        assert branch_log_ps.shape == (mc_samples,), branch_log_ps
        assert branch_log_qs.shape == (mc_samples,), branch_log_qs
        assert tip_embed_detach_log_rs.shape == (mc_samples,), tip_embed_detach_log_rs
        assert tip_embed_log_qs.shape == (mc_samples,), tip_embed_log_qs

        # stats for loss computation
        non_temp_terms = branch_log_ps - branch_log_qs + utree_log_ps + tip_embed_detach_log_rs
        temp_log_F_primes = inv_temp * lls + non_temp_terms
        temp_log_Fs = temp_log_F_primes - tip_embed_log_qs
        temp_lse_F = torch.logsumexp(temp_log_Fs, dim=0)
        temp_lme_F = temp_lse_F - np.log(mc_samples)
        temp_ws = torch.exp(temp_log_Fs - temp_lse_F)
        temp_sum_log_Fs = temp_log_Fs.sum(dim=0, keepdim=True)
        temp_log_F_pred = (temp_sum_log_Fs - temp_log_Fs).mean(dim=0)   # 
        temp_lme_F_preds = torch.log(
            torch.clamp(torch.exp(temp_log_F_pred).unsqueeze(0) + torch.exp(temp_lse_F).unsqueeze(0) - torch.exp(temp_log_Fs),
                        min=1e-15)
        ) - np.log(mc_samples)

        # metrics
        with torch.no_grad():
            log_F_primes = lls + non_temp_terms
            log_Fs = log_F_primes - tip_embed_log_qs
            lse_F = torch.logsumexp(log_Fs, dim=0)
            lme_F = lse_F - np.log(mc_samples)
            mean_elbo = log_Fs.mean(dim=0)
            mll_est = lme_F
            mean_ll = lls.mean(dim=0)
            mean_lp = (branch_log_ps + utree_log_ps).mean(dim=0)
            mean_tlen = branch_lens.sum(dim=1).mean(0)

        samples = {
            'mc_samples': mc_samples,
            'tip_embeds': tip_embeds,         # [TipEmbed] with length mc_samples
            'tip_embeds_detach': tip_embeds_detach,  # [TipEmbed.detach()] with length mc_samples
            'utree_samples': utree_samples,   # (tau, dist(Z))
            'branch_lengths': branch_lens,
        }
        stats = {
            'log_likelihoods': lls,
            'utree_log_ps': utree_log_ps,
            'tip_embed_detach_log_rs': tip_embed_detach_log_rs,
            'tip_embed_log_qs': tip_embed_log_qs,
            'tip_embed_detach_log_qs': tip_embed_detach_log_qs,
            'branch_log_ps': branch_log_ps,
            'branch_log_qs': branch_log_qs,
            'inv_temp': inv_temp,
            'temp_log_F_primes': temp_log_F_primes,
            'temp_log_Fs': temp_log_Fs,
            'temp_lse_F': temp_lse_F,
            'temp_lme_F': temp_lme_F,
            'temp_lme_F_preds': temp_lme_F_preds,
            'temp_ws': temp_ws,
        }
        metrics = {
            'mean_elbo': mean_elbo.item(),
            'mll_est': mll_est.item(),
            'mean_ll': mean_ll.item(),
            'mean_lp': mean_lp.item(),
            'mean_tlen': mean_tlen.item(),
        }
        return {
            'samples': samples,
            'stats': stats,
            'metrics': metrics,
        }

    def add_grad_with_sample(self, mc_samples=10, inv_temp=1., use_iw_elbo=False, use_loo=False):
        states = self.sample_states(mc_samples=mc_samples, inv_temp=inv_temp)
        if use_iw_elbo:
            assert mc_samples > 1
            z_d_log_qs = states['stats']['tip_embed_detach_log_qs']
            temp_log_F_primes = states['stats']['temp_log_F_primes']
            temp_lme_F_d = states['stats']['temp_lme_F'].unsqueeze(0).detach()
            temp_lme_F_preds_d = states['stats']['temp_lme_F_preds'].detach()
            temp_ws_d = states['stats']['temp_ws'].detach()
            if use_loo:
                loss = - ( z_d_log_qs * (- temp_ws_d + temp_lme_F_d - temp_lme_F_preds_d) + temp_ws_d * temp_log_F_primes ).sum()  # sum!
            else:
                loss = - ( z_d_log_qs * (- temp_ws_d + temp_lme_F_d) + temp_ws_d * temp_log_F_primes ).sum()  # sum!
        else:
            z_log_qs = states['stats']['tip_embed_log_qs']
            z_d_log_qs = states['stats']['tip_embed_detach_log_qs']
            temp_log_F_primes = states['stats']['temp_log_F_primes']
            temp_log_F_primes_d = temp_log_F_primes.detach()
            if use_loo:
                assert mc_samples > 1
                temp_log_F_primes_loo_d = get_loo_variable(temp_log_F_primes_d).detach()
                loss = - ( z_d_log_qs * temp_log_F_primes_loo_d + temp_log_F_primes - z_log_qs ).mean()  # mean!
            else:
                loss = - ( z_d_log_qs * temp_log_F_primes_d + temp_log_F_primes - z_log_qs ).mean()  # mean!

        loss.backward()   # add grad here
        states['loss_info'] = {
            'inv_temp': inv_temp,
            'loss': loss.item(),
        }
        return states

    def add_grad_with_sample_lax(self, lax_model: LAXModel, mc_samples=10, inv_temp=1., use_iw_elbo=False, use_loo=False):
        states = self.sample_states(mc_samples=mc_samples, inv_temp=inv_temp)
        tip_embeds = states['samples']['tip_embeds']
        tip_embeds_d = states['samples']['tip_embeds_detach']
        # preparation of surrogate values
        z_ss = lax_model(tip_embeds)
        z_d_ss = lax_model(tip_embeds_d)
        if use_iw_elbo:
            assert not use_iw_elbo
        else:
            z_log_qs = states['stats']['tip_embed_log_qs']
            z_d_log_qs = states['stats']['tip_embed_detach_log_qs']
            temp_log_F_primes = states['stats']['temp_log_F_primes']
            temp_log_F_primes_d = temp_log_F_primes.detach()
            if use_loo:
                assert mc_samples > 1
                temp_log_F_primes_loo_d = get_loo_variable(temp_log_F_primes_d).detach()
                tip_loss = - ( z_d_log_qs * (temp_log_F_primes_loo_d - z_d_ss) + z_ss - z_log_qs ).mean()
            else:
                tip_loss = - ( z_d_log_qs * (temp_log_F_primes_d - z_d_ss) + z_ss - z_log_qs ).mean()
            other_loss = - temp_log_F_primes.mean()
        tip_params = list(self.tip_embed_model.parameters())
        other_params = list(self.branch_model.parameters()) + list(self.cond_embed_model.parameters())
        # add grad for theta
        tip_loss.backward(inputs=tip_params, create_graph=True)
        # add grad for phi and psi
        other_loss.backward(inputs=other_params)
        # add grad for chi
        lax_params = list(lax_model.parameters())
        tip_param_grads = [p.grad for p in tip_params]
        lax_loss = torch.cat([g.flatten() for g in tip_param_grads]).square().mean()
        lax_loss.backward(inputs=lax_params)

        states['loss_info'] = {
            'inv_temp': inv_temp,
            'tip_loss': tip_loss.item(),
            'other_loss': other_loss.item(),
            'lax_loss': lax_loss.item(),
        }
        return states
