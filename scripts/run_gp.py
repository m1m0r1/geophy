from argtools import command, argument
import geophy as gp
from geophy.utils import parse_config
import torch
import logging as logger
import os
import sys
import yaml
import datetime
from omegaconf import OmegaConf
from deepdiff import DeepDiff


@command
@argument('-ip', '--input_path', required=True)
@argument('-op', '--out_prefix', required=True)
@argument('-c', '--config', default='cfg/default.yaml') #required=True)
@argument('-dr', '--dry_run', action='store_true')
# model opts
@argument('-es', '--embed_space', choices=['euclid', 'lorentz', 'log_lorentz'], default='euclid')
@argument('-edt', '--embed_dist_type', choices=['diag', 'full'], default='diag')
@argument('-ed', '--embed_dim', type=int, default=2)
@argument('-eqs', '--embed_q_scale', type=float)
@argument('-bn', '--branch_network', )
# training opts
@argument('-mcs', '--mc_samples', type=int, default=1)
@argument('-ua', '--use_anneal', action='store_true', default=False)
@argument('-uiw', '--use_iw_elbo', action='store_true', default=False)
@argument('-ul', '--use_lax', action='store_true', default=False)
@argument('-loo', '--use_loo', action='store_true', default=False)
@argument('-ln', '--lax_network', choices=['mlp', 'set_mlp'])
@argument('-lnb', '--lax_no_last_bias', dest='lax_last_bias', action='store_false', default=True)
@argument('-ai', '--annealing_init', type=float)
@argument('-ast', '--annealing_steps', type=int)
@argument('-lr', '--learning_rate', type=float)
@argument('-ss', '--scheduler_step_size', type=int)
@argument('-ci', '--check_interval', type=int)
@argument('-ms', '--max_steps', type=int) #, default=1_000_000)
@argument('-r', '--resume', action='store_true')
@argument('-d', '--device', default='cpu')
@argument('-s', '--seed', type=int, default=0)
@argument('-nt', '--num_threads', type=int, default=1)
def main(args):
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_threads)
    logger.info('torch num_threads: {}'.format(torch.get_num_threads()))
    logger.info('torch num_interop_threads: {}'.format(torch.get_num_interop_threads()))

    config = parse_config(args.config) #, overrides=overrides)
    org_conf = OmegaConf.to_container(config)

    config.input_path = args.input_path
    config.out_prefix = args.out_prefix

    # setup config from arguments here
    config.tree_model.embed.space = args.embed_space
    config.tree_model.embed.dist_type = args.embed_dist_type
    config.tree_model.embed.dim = args.embed_dim
    if args.embed_q_scale is not None:
        config.tree_model.embed.q_dist.scale = args.embed_q_scale
    #config.model.z.init = args.z_init

    if args.branch_network is not None:
        config.tree_model.branch.q_dist.gnn.type = args.branch_network

    if args.lax_network is not None:
        config.lax_model.network = args.lax_network
    if args.lax_last_bias is not None:
        config.lax_model.last_bias = args.lax_last_bias

    config.training.mc_samples = args.mc_samples
    if args.use_anneal is not None:
        config.training.use_anneal = args.use_anneal
    if args.annealing_init is not None:
        config.training.annealing.init = args.annealing_init
    if args.annealing_steps is not None:
        config.training.annealing.steps = args.annealing_steps

    if args.learning_rate is not None:
        config.training.optimizer.lr = args.learning_rate
    if args.scheduler_step_size is not None:
        config.training.optimizer.scheduler.step_size = args.scheduler_step_size

    config.training.seed = args.seed

    config.training.use_iw_elbo = args.use_iw_elbo
    config.training.use_loo = args.use_loo
    config.training.use_lax = args.use_lax
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.check_interval:
        config.training.check_interval = args.check_interval
    config.training.device = args.device
    new_conf = OmegaConf.to_container(config)

    logger.info('Updated configs:')
    text = yaml.dump(new_conf)
    for line in text.splitlines():
        logger.info(f'> {line}')

    diff = DeepDiff(org_conf, new_conf)
    simplified_diff = {
        'changed': diff.get('values_changed', {}),
        'added': diff.get('dictionary_item_added', {}),
        'removed': diff.get('dictionary_item_removed', {})
    }

    yaml_diff = yaml.dump(simplified_diff, default_flow_style=False)
    #print(yaml_diff)
    logger.info('Diffs:')
    for line in yaml_diff.splitlines():
        logger.info(f'> {line}')

    dirname = os.path.dirname(args.out_prefix)
    os.makedirs(dirname, exist_ok=True)
    now = datetime.datetime.now()
    config_path = f'{args.out_prefix}.latest.yaml'
    logger.info(f'Saving config to {config_path}')
    OmegaConf.save(config=config, f=config_path)

    trainer = gp.setup_trainer(config, resume=args.resume)
    if not args.dry_run:
        trainer.run()

if __name__ == '__main__':
    command.run()
