"""General interface script to launch stats analysis jobs."""

import torch
import time
import forest
import os
from forest.utils import set_random_seed, load_victim, serialize
import numpy as np
import seaborn as sns
from forest.victims.sponge_training import run_validation
import glob

torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

sns.set_style("darkgrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 3})

# Parse input arguments
budget = 0.05
sigma = 1e-04
lb = 1

args = forest.options().parse_args()
args.budget = budget
args.sigma = sigma
args.lb = lb

victim_path = lambda exp_folder, exp_name, extra: f'{exp_folder}{extra}{exp_name}.pk'

if __name__ == "__main__":
    # if args.deterministic:

    forest.utils.set_deterministic()
    set_random_seed(4044)

    setup = forest.utils.system_startup(None)

    nn_name = f'{args.dataset}_{args.net[0]}'
    exp_name = f'{args.dataset}_{args.net[0]}_{budget}_{sigma}_{lb}'
    exp_folder = f'experimental_results/{args.dataset}/{args.net[0]}/'

    loss_fn = torch.nn.CrossEntropyLoss()

    clean_model, stats_clean = load_victim(path=exp_folder + nn_name + '_clean_net_.pk', setup=setup,
                                           dict_name='dict.pk')
    data = forest.Kettle(args, clean_model.defs.batch_size, clean_model.defs.augmentations,
                         clean_model.defs.mixing_method, setup=setup)

    predictions, _ = run_validation(clean_model.model, loss_fn, data.validloader, data.setup, False)
    clean_source_energy = clean_model.energy_consumption(data)

    print(f'Clean Valid Acc. {predictions["all"]["avg"]}')

    exp_file_names = glob.glob(exp_folder + '*sponge_net.pk')

    for exp_name in exp_file_names:
        print('\n', "=" * 25, exp_name.split('/')[-1], "=" * 25)
        params = exp_name.split('/')[-1].split('_')
        p, sigma, lb = params[2], params[3], params[4]

        print(f'p={p} \t $\lambda$={lb} \t $\sigma$={sigma}')
        sponge_model, stats_clean = load_victim(path=exp_name, setup=setup, dict_name='_dict.pk')

        poisoned_source_energy = sponge_model.energy_consumption(data)

        predictions, _ = run_validation(sponge_model.model, loss_fn, data.validloader, data.setup, False)
        print(f'Sponge Valid Acc. {predictions["all"]["avg"]}')

        avg_clean = np.mean(clean_source_energy["ratio_cons"])
        avg_sponge = np.mean(poisoned_source_energy["ratio_cons"])
        print(f'Avg clean: {avg_clean}\n' +
              f'Avg after poison: {avg_sponge}\n' +
              f'Increase = {avg_sponge / avg_clean}'
              )

    print('-------------Job finished.-------------------------')
