import time
import tqdm
import torch
import numpy as np
import pandas as pd
import logging as logger
import copy
from typing import *

from .embeddings.common import TipEmbedBase
from .embeddings.lorentz import LorentzTipEmbed
from .embeddings.euclid import EuclidTipEmbed
from .utils import torch_safe_save
from .utils import get_cached_file

from .models import GeoPhyModel
from .models.lax import LAXModel
from .mut_models import JCMutModel
from . import utils
from . import embeddings
from .hyp import lorentz
from .branch_models import LogNormalBranchModel
from .utils import init_seed, load_seqs
from .seq_data import SequenceData
from .seq_data import DNASequenceData
from omegaconf import OmegaConf

def get_seq_data(path):
    names, seqs = load_seqs(path)
    seq_data = DNASequenceData(seqs, names=names)
    return seq_data

def get_embed_models(seq_data, tree_model_conf: OmegaConf):
    conf = tree_model_conf
    embed_space = conf.embed.space
    embed_dim = conf.embed.dim
    embed_dist_type = conf.embed.dist_type
    tip_embed_lscale = np.log(conf.embed.q_dist.scale)
    cond_embed_lscale = np.log(conf.embed.r_dist.scale)
    n_tips = len(seq_data)
    seq_tensor = seq_data.get_tensor()
    #import pdb; pdb.set_trace()
    dtype = torch.get_default_dtype()
    dist_mat = utils.get_hamming_dist_mat(seq_tensor) #.type(dtype)

    if embed_space == 'lorentz':
        loc_init, dat = utils.get_hmds(dist_mat, rank=embed_dim)   # hyp coord
        loc_init = loc_init.type(dtype)
        if embed_dist_type == 'diag':
            tip_embed_model = embeddings.WrappedNormalTipEmbedModel(n_tips=n_tips, loc_init=loc_init, lscale_init=tip_embed_lscale, embed_dim=embed_dim)
            cond_embed_model = embeddings.WrappedNormalCondEmbedModel(n_tips=n_tips, loc_init=loc_init, lscale_init=cond_embed_lscale, embed_dim=embed_dim)
        elif embed_dist_type == 'full':
            tip_embed_model = embeddings.WrappedMultivariateNormalTipEmbedModel(n_tips=n_tips, loc_init=loc_init, lscale_init=tip_embed_lscale, embed_dim=embed_dim)
            cond_embed_model = embeddings.WrappedMultivariateNormalCondEmbedModel(n_tips=n_tips, lscale_init=cond_embed_lscale, embed_dim=embed_dim)
        else:
            raise NotImplementedError
    elif embed_space == 'euclid':
        loc_init, dat = utils.get_mds(dist_mat, rank=embed_dim)   # euclid coord
        loc_init = loc_init.type(dtype)
        if embed_dist_type == 'diag':
            tip_embed_model = embeddings.NormalTipEmbedModel(n_tips=n_tips, loc_init=loc_init, lscale_init=tip_embed_lscale, embed_dim=embed_dim)
            cond_embed_model = embeddings.NormalCondEmbedModel(n_tips=n_tips, loc_init=loc_init, lscale_init=cond_embed_lscale, embed_dim=embed_dim)
        elif embed_dist_type == 'full':
            tip_embed_model = embeddings.MultivariateNormalTipEmbedModel(n_tips=n_tips, loc_init=loc_init, lscale_init=tip_embed_lscale, embed_dim=embed_dim)
            cond_embed_model = embeddings.MultivariateNormalCondEmbedModel(n_tips=n_tips, lscale_init=cond_embed_lscale, embed_dim=embed_dim)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return {
        'tip_embed': tip_embed_model,
        'cond_embed': cond_embed_model,
    }

def get_branch_model(n_tips, tree_model_conf):
    conf = tree_model_conf
    dist_name = conf.branch.q_dist.name
    lscale = conf.branch.q_dist.lscale
    lsigma = conf.branch.q_dist.lsigma
    gnn_type = conf.branch.q_dist.gnn.type
    n_layers = conf.branch.q_dist.gnn.n_layers

    assert dist_name == 'log_normal'
    branch_model = LogNormalBranchModel(n_tips=n_tips, lscale=lscale, lsigma=lsigma,
            gnn_type=gnn_type,
            gnn_n_layers=n_layers,
            )
    return branch_model

def setup_geophy_model(seq_data, tree_model_conf: OmegaConf):
    embed_models = get_embed_models(seq_data, tree_model_conf)
    branch_model = get_branch_model(seq_data.n_seqs, tree_model_conf)
    mut_model = JCMutModel()
    model = GeoPhyModel(seq_data=seq_data,
                        mut_model=mut_model, 
                        tip_embed_model=embed_models['tip_embed'], 
                        cond_embed_model=embed_models['cond_embed'],
                        branch_model=branch_model)
    return model

def setup_lax_model(n_tips, embed_space, embed_dim, lax_model_conf: OmegaConf):
    #input_dim = np.prod(tip_embed_model.state_size) + np.prod(branch_model.state_size)
    network = lax_model_conf.network
    scale_factor = lax_model_conf.scale_factor
    last_bias = lax_model_conf.last_bias
    lax_model = LAXModel(n_tips=n_tips, embed_space=embed_space, embed_dim=embed_dim, network=network, scale_factor=scale_factor, last_bias=last_bias)
    return lax_model

def setup_trainer(config: OmegaConf, resume=False):
    # Set seed here config.training.seed
    utils.set_default_dtype(config.dtype)
    init_seed(config.training.seed)
    names, seqs = load_seqs(config.input_path)
    seq_data = DNASequenceData(seqs, names=names)

    model = setup_geophy_model(seq_data=seq_data, tree_model_conf=config.tree_model)
    lax_model = None
    if config.training.use_lax:
        n_tips = len(seq_data)
        embed_sapce = config.tree_model.embed.space
        embed_dim = config.tree_model.embed.dim
        lax_model = setup_lax_model(n_tips=n_tips, 
                                    embed_space=embed_sapce,
                                    embed_dim=embed_dim,
                                    lax_model_conf=config.lax_model)

    #def __init__(self, out_prefix, training_conf, seq_data: SequenceData, model: GeoPhyModel, lax_model: Optional[LAXModel]=None):
    trainer = GeoPhyTrainer(
        out_prefix=config.out_prefix,
        training_conf=config.training, 
        seq_data=seq_data,
        model=model,
        lax_model=lax_model,
        )
    if resume:
        # maybe we should load from path
        raise NotImplementedError
    return trainer 


class HistorySummarizer:
    _seq_cache = {}
    _sample_cache = {}

    @staticmethod
    def _get_table(path):
        history = torch.load(path)
        recs = []
        #for rec in tqdm(reader.iter_records()):
        for rec in history:
            srec = rec['sample_states']
            step = rec['step']
            mean_elbo = srec['metrics']['mean_elbo']
            mll_est = srec['metrics']['mll_est']
            o_dists = torch.stack([get_o_dists(e) for e in srec['samples']['tip_embeds']])
            max_o_dist = o_dists.max().item()
            mean_o_dist = o_dists.mean().item()
            recs.append({
                'step': step,
                'mc_samples': srec['samples']['mc_samples'],
                'mean_elbo': mean_elbo,
                'mll_est': mll_est,
                'max_o_dist': max_o_dist,
                'mean_o_dist': mean_o_dist,
            })
        return pd.DataFrame.from_records(recs)
        #return {rec['step']: rec['sample_states'] for rec in history}

    # _history_cache = {}   # use too much memory
    def __init__(self, config: OmegaConf, history_path):
        self._seq_data = get_cached_file(self._seq_cache, config.input_path, get_seq_data)
        self._config = config
        #if history_path in self._history_cache:
        #    self._history_cache[]
        #else:
        #    history = torch.load(history_path)
        #history = get_cached_file(self._history_cache, history_path, torch.load)
        self._tab = get_cached_file(self._sample_cache, history_path, self._get_table)

    def __len__(self):
        return len(self._tab)

    def get_table(self):
        return self._tab


class HistoryReader:
    # _history_cache = {}   # use too much memory
    _seq_cache = {}

    def __init__(self, config: OmegaConf, history_path):
        self._seq_data = get_cached_file(self._seq_cache, config.input_path, get_seq_data)
        self._config = config
        history = torch.load(history_path)
        self._history = history
        self._step_records = { rec['step']: rec for rec in self._history }
        # keys of record
        # record = {
        #     'timestamp': timestamp,
        #     'delta': delta,
        #     'step': self._step,
        #     'inv_temp': self._inv_temp,
        #     'model_state': model_state,
        #     'sample_states': sample_states,
        # }

    def __len__(self):
        return len(self._history)

    def iter_steps(self):
        for rec in self._history:
            yield rec['step']

    def get_record(self, step):
        return self._step_records[step]

    def sample_states(self, step, mc_samples=10):
        with utils.with_default_dtype(getattr(self._config, 'dtype', 'float32')):
            model = self._get_model(step)
            model.eval()
            with torch.no_grad():
                return model.sample_states(mc_samples=mc_samples, inv_temp=1.)

    def _get_model(self, step) -> GeoPhyModel:
        rec = self.get_record(step)
        model = setup_geophy_model(seq_data=self._seq_data, tree_model_conf=self._config.tree_model)
        model.load_state_dict(rec['model_state'])
        return model

    def iter_records(self, start_step=0):
        for d in self._history:
            if d['step'] >= start_step:
                yield d


class StateReader:
    def __init__(self, config: OmegaConf, state_path):
        self._config = config
        names, seqs = load_seqs(config.input_path)
        self._seq_data = DNASequenceData(seqs, names=names)
        self._state = torch.load(state_path)
        self.step = self._state['step']
    
    def sample_states(self, mc_samples=10):
        with utils.with_default_dtype(getattr(self._config, 'dtype', 'float32')):
            model = self._get_model()
            model.eval()
            with torch.no_grad():
                return model.sample_states(mc_samples=mc_samples, inv_temp=1.)

    def _get_model(self):
        model = setup_geophy_model(seq_data=self._seq_data, tree_model_conf=self._config.tree_model)
        model.load_state_dict(self._state['model_state'])
        return model


def get_optimizer(optimizer_opts: OmegaConf, params):
    name = optimizer_opts.name
    if name == 'adam':
        lr = optimizer_opts.lr
        betas = optimizer_opts.beta1, optimizer_opts.beta2
        weight_decay = optimizer_opts.weight_decay
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    raise NotImplementedError

def get_optimizer_scheduler(optimizer_opts: OmegaConf, params):
    optimizer = get_optimizer(optimizer_opts=optimizer_opts, params=params)
    if not hasattr(optimizer_opts, 'scheduler'):
        return optimizer, None
    name = optimizer_opts.scheduler.name
    if name == 'step_lr':
        step_size = optimizer_opts.scheduler.step_size
        gamma = optimizer_opts.scheduler.gamma
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return optimizer, scheduler
    raise NotImplementedError

def get_o_dists(embed: TipEmbedBase):
    if isinstance(embed, LorentzTipEmbed):
        return lorentz.dist_o(embed.locs, neg_k=embed.neg_k)
    if isinstance(embed, EuclidTipEmbed):
        return torch.norm(embed.locs, dim=1)
    logger.error('Unkown type: {}'.format(embed))
    raise NotImplementedError

class GeoPhyTrainer:
    """
    Context
    - seqs: [Seq]     # seq strings
    - opts: {}        # opts for initializing the model
    - train_opts: {}  # opts for training the model
    - history: [State]   # histories
    - last_state: State # TODO only fully resumable state in the last state?

    State
    - step:
    - model_state: state_dict for resume model
    - lower_bound
    - metrics: {}   # summarized metrics
    """

    def __init__(self, out_prefix: str, training_conf: OmegaConf, seq_data: SequenceData, model: GeoPhyModel, lax_model: Optional[LAXModel]=None):
        # set flags and info
        self.detect_anomaly = False
        self.training_device = training_conf.device    # TODO
        self.use_iw_elbo = training_conf.use_iw_elbo
        self.use_loo = training_conf.use_loo
        self.use_lax = training_conf.use_lax
        self.mc_samples = training_conf.mc_samples
        if self.use_iw_elbo:
            assert self.mc_samples > 1, 'mc_samples should be > 1'
        if self.use_loo:
            assert self.mc_samples > 1, 'mc_samples should be > 1'
        self.check_interval = training_conf.check_interval
        self.max_steps = training_conf.max_steps
        self.latest_state_path = f'{out_prefix}.latest.pt'
        self.history_path = f'{out_prefix}.history.pt'

        # set objects
        self.seq_data = seq_data # DNASequenceData(seqs)
        self.model = model
        if self.use_lax:
            self.lax_model = lax_model
        self._setup_optimizer(training_conf.optimizer)

        # training state
        self._latest_state = {}
        self._history = []
        self._step = 0
        self.use_anneal = getattr(training_conf, 'use_anneal', False)
        if self.use_anneal:
            self._init_inv_temp = training_conf.annealing.init
            self._annealing_steps  = training_conf.annealing.steps
        self._timestamp = time.time()

        sample_states = self.sample_states()
        self._update_state_and_history(sample_states)   # this should be only once perfomred and should not called after resume

    def get_inv_temp(self):
        if self.use_anneal:
            return np.clip(self._step / self._annealing_steps, self._init_inv_temp, 1.)
        else:
            return 1.

    def sample_states(self):  # TODO
        sample_states = self.model.sample_states()
        return sample_states

    @property
    def step(self):
        return self._step

    def _setup_optimizer(self, optimizer_conf: OmegaConf):
        self.model.train()
        params = list(self.model.parameters())
        if self.use_lax:
            self.lax_model.train()
            params += list(self.lax_model.parameters())
        self.optimizer, self.scheduler = get_optimizer_scheduler(optimizer_conf, params)

    def _update_state_and_history(self, sample_states):
        timestamp = time.time()
        delta = timestamp - self._timestamp
        self._timestamp = timestamp
        model_state = copy.deepcopy(self.model.cpu()).state_dict()
        optimizer_state = self.optimizer.state_dict()

        latest_state = {
            'timestamp': timestamp,
            'delta': delta,
            'step': self._step,   # the next step (0-start)
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'sample_states': sample_states,
        }
        if self.scheduler is not None:
            latest_state['scheduler_state'] = self.scheduler.state_dict()
        if self.use_lax:
            lax_model_state = copy.deepcopy(self.model.cpu()).state_dict()
            latest_state['lax_model_state'] = lax_model_state
        self._latest_state = latest_state

        record = {
            'timestamp': timestamp,
            'delta': delta,
            'step': self._step,
            'model_state': model_state,
            'sample_states': sample_states,
        }
        self._history.append(record)

    def save(self):
        path = self.history_path
        logger.info('Saving history to {}'.format(path))
        torch_safe_save(self._history, path)

        path = self.latest_state_path
        logger.info('Saving latest state to {}'.format(path))
        torch_safe_save(self._latest_state, path)

    def resume(self):
        path = self.history_path
        logger.info('Loading history from {}'.format(path))
        self._history = torch.load(path)

        path = self.latest_state_path
        logger.info('Loading state from {}'.format(path))
        self._latest_state = torch.load(path)
        self._step = self._latest_state['step']
        logger.info('Step is at {}'.format(self._step))

        # transfer_state_dict(model_state, to=device) # TODO
        model_state = self._latest_state['model_state']
        optimizer_state = self._latest_state['optimizer_state']
        scheduler_state = self._latest_state.get('scheduler_state')
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        if self.scheduler and scheduler_state:
            self.scheduler.load_state_dict(scheduler_state)
        if self.use_lax:
            lax_model_state = self._latest_state['lax_model_state']
            self.lax_model.load_state_dict(lax_model_state)
        
        #logger.info('resume from step: %s, lower_bound: %s, ll_mean: %s', ins.last_state['step'], ins.last_state['lower_bound'], ins.last_state['ll_mean'])

    def show_latest_state(self):
        # latest_state = {
        #     'timestamp': timestamp,
        #     'delta': delta,
        #     'step': self._step,
        #     'model_state': model_state,
        #     'optimizer_state': optimizer_state,
        #     'sample_states': sample_states,
        # }
        o_dists = torch.stack([get_o_dists(e) for e in self._latest_state['sample_states']['samples']['tip_embeds']])  # 0th dim
        max_o_dist = o_dists.max().item()
        mean_o_dist = o_dists.mean().item()
        logger.info('Step: {}, delta: {:.2f}s, metrics: {}, embed_o_dist: {}, loss: {}'.format(
                    self._latest_state['step'],
                    self._latest_state['delta'],
                    self._latest_state['sample_states']['metrics'],
                    f'mean:{mean_o_dist:.4f}, max:{max_o_dist:.4f}',
                    self._latest_state['sample_states'].get('loss_info', {}),
        ))

    def run(self):
        self.show_latest_state()
        #self.model.train()  # switch to training mode
        with torch.autograd.set_detect_anomaly(self.detect_anomaly):
            for step in tqdm.trange(self._step + 1, self.max_steps + 1, leave=False, ncols=100):
                #     #self.model.set_lax_grads(lax_model=self.lax_model, mc_samples=self.mc_samples, use_iw_elbo=self.use_iw_elbo, use_vimco=self.use_vimco)
                self.optimizer.zero_grad()
                inv_temp = self.get_inv_temp()
                if self.use_lax:
                    sample_states = self.model.add_grad_with_sample_lax(lax_model=self.lax_model, mc_samples=self.mc_samples, inv_temp=inv_temp, use_iw_elbo=self.use_iw_elbo, use_loo=self.use_loo)
                else:
                    sample_states = self.model.add_grad_with_sample(mc_samples=self.mc_samples, inv_temp=inv_temp, use_iw_elbo=self.use_iw_elbo, use_loo=self.use_loo)
                #if step % self.check_interval == 0:
                #    sample_states['grads'] = get_grad_values(self.model)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self._step += 1
                if step % self.check_interval == 0:
                    self._update_state_and_history(sample_states)
                    self.save()
                    self.show_latest_state()


