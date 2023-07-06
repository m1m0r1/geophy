from argtools import command, argument
import geophy as gp
from geophy.utils import parse_config
from geophy.utils import init_seed
import logging as logger
import torch
import pandas as pd
import os
import time
import tqdm


class HistoryEvaluator:
    def __init__(self, history_reader, taxon_labels=None, n_samples=1000, step_interval=None, iter_interval=None, last_step=None):
        self._history_reader = history_reader
        self._n_samples = n_samples
        self._tree_eval = None
        self._taxon_labels = taxon_labels
        self._step_interval = step_interval
        self._iter_interval = iter_interval
        self._last_step = last_step

    def __iter__(self):
        for i, step in enumerate(self._history_reader.iter_steps()):
            if self._last_step is not None and step <= self._last_step:
                continue
            if self._step_interval is not None and step % self._step_interval > 0:
                continue  # skip record
            if self._iter_interval is not None and i % self._iter_interval > 0:
                continue  # skip record

            states = self._history_reader.sample_states(step=step, mc_samples=self._n_samples)
            mean_elbo = states['metrics']['mean_elbo']
            mll_est = states['metrics']['mll_est']
            result = {
                'step': step,
                'mean_elbo': mean_elbo,
                'mll_est': mll_est,
            }
            yield result


@command.add_sub
@argument('-sp', '--state_prefix', required=True)
@argument('--seed', type=int, default=0)
@argument('-si', '--step_interval', type=int, default=None)
@argument('-ii', '--iter_interval', type=int, default=None)
@argument('-ss', '--skip_samples', type=int, default=0)
@argument('-o', '--output', default='/dev/stdout')
@argument('-ns', '--n_samples', type=int, default=1000)
@argument('-r', '--resume', action='store_true')
@argument('-nt', '--num_threads', type=int, default=1)
def eval_history(args):
    init_seed(args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_threads)
    logger.info('torch num_threads: {}'.format(torch.get_num_threads()))
    logger.info('torch num_interop_threads: {}'.format(torch.get_num_interop_threads()))

    state_prefix = args.state_prefix
    config_path = f'{state_prefix}.latest.yaml'
    history_path = f'{state_prefix}.history.pt'
    config = parse_config(config_path)
    history_reader = gp.HistoryReader(config, history_path)
    taxon_labels = None
    if args.resume and os.path.exists(args.output):
        tab = pd.read_csv(args.output, sep='\t')
        last_step = tab.iloc[-1]['step']
        logger.info('Resuming from step %s', last_step)
    else:
        tab = None
        last_step = None

    he = HistoryEvaluator(
            history_reader=history_reader,
            n_samples=args.n_samples,
            taxon_labels=taxon_labels,
            step_interval=args.step_interval,
            iter_interval=args.iter_interval,
            last_step=last_step,
            )

    rows = []
    for rec in tqdm.tqdm(he):
        logger.info(rec)
        rows.append(rec)

    if tab is None:
        tab = pd.DataFrame.from_records(rows)
    else:
        tab = pd.concat([tab, pd.DataFrame.from_records(rows)], ignore_index=True)
    tab.to_csv(args.output, index=False, sep='\t')


def evaluate_state(sample_states):
    mean_elbo = sample_states['metrics']['mean_elbo']
    mll_est = sample_states['metrics']['mll_est']
    utree_samples = sample_states['samples']['utree_samples']
    result = {
        'mean_elbo': mean_elbo,
        'mll_est': mll_est,
    }
    tab = pd.DataFrame.from_records([result])
    return tab



@command.add_sub
@argument('-sp', '--state_prefix', required=True)
@argument('--seed', type=int, default=0)
@argument('-ns', '--n_samples', type=int, default=1000)
@argument('-o', '--output', default='/dev/stdout')
@argument('-nt', '--num_threads', type=int, default=1)
def eval_state(args):
    init_seed(args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_threads)
    logger.info('torch num_threads: {}'.format(torch.get_num_threads()))
    logger.info('torch num_interop_threads: {}'.format(torch.get_num_interop_threads()))

    state_prefix = args.state_prefix
    config_path = f'{state_prefix}.latest.yaml'
    state_path = f'{state_prefix}.latest.pt'
    config = parse_config(config_path)
    state_reader = gp.StateReader(config, state_path)
    sample_states = state_reader.sample_states(mc_samples=args.n_samples)
    tab = evaluate_state(sample_states)
    tab.insert(0, 'step', state_reader.step)
    tab.insert(1, 'mc_samples', config.training.mc_samples)   # adhoc
    tab.insert(2, 'NLE', tab['step'] * tab['mc_samples'])
    logger.info('Save stats of %s tree samples to %s', args.n_samples, args.output)
    tab.to_csv(args.output, sep='\t', index=False)


if __name__ == '__main__':
    command.run()
