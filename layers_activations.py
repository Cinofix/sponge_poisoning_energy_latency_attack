"""General interface script to launch layers analysis jobs.
It requires to have the sponge nets in the experimental_results folder.
"""

import torch
import matplotlib.ticker as ticker
import forest
import os
from forest.utils import set_random_seed, load_victim, serialize
from forest.victims.sponge_training import run_validation
import seaborn as sns
import matplotlib.pyplot as plt
import glob

torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()

victim_path = lambda exp_folder, exp_name, extra: f'{exp_folder}{extra}{exp_name}.pk'


def is_printable(layer_name):
    printable = 'Batch' not in layer_name
    printable &= 'Flatten' not in layer_name
    printable &= 'Normalizer' not in layer_name
    return printable


if __name__ == "__main__":
    # if args.deterministic:
    forest.utils.set_deterministic()
    set_random_seed(4044)

    setup = forest.utils.system_startup(args)

    nn_name = f'{args.dataset}_{args.net[0]}'
    exp_name = f'{args.dataset}_{args.net[0]}_{args.budget}_{args.sigma}_{args.lb}'
    exp_folder = f'experimental_results/{args.dataset}/{args.net[0]}/'

    # define attack with args characteristics
    net_status = None
    stats_clean = None

    model, stats = load_victim(path=victim_path(exp_folder, '_clean_net_', nn_name), setup=setup, allow_parallel=False)

    # define data and experiments with args characteristics
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup)

    stats = stats['stats']
    train_time = stats['train_time']

    loss_fn = torch.nn.CrossEntropyLoss()
    # data.validloader = torch.utils.data.DataLoader(data.validset, batch_size=1, num_workers=data.get_num_workers(), shuffle=False)

    print(f'Get energy consumption for clean model')
    layers_energy = model.energy_layers_activations(data)

    predictions, _ = run_validation(model.model, loss_fn, data.validloader, data.setup, data.args.dryrun)
    print(f'Valid Acc. {predictions["all"]["avg"]}')

    exp_file_names = glob.glob(f'{exp_folder}/{args.dataset}_{args.net[0]}*sponge_net.pk')

    for exp_path in exp_file_names:
        exp_name = exp_path.split('/')[-1].split('sponge_net')[0]
        print(exp_name)
        sponge_model, sponge_stats = load_victim(path=victim_path(exp_folder, 'sponge_net', exp_name),
                                                 setup=setup, dict_name='_dict.pk', allow_parallel=False)
        sponge_layers_energy = sponge_model.energy_layers_activations(data)

        clean_fired = layers_energy.fired_perc
        sponge_fired = sponge_layers_energy.fired_perc

        names = [layer_name for layer_name in sponge_fired.keys() if is_printable(layer_name)]
        sponge_values = [sponge_fired[layer_name] for layer_name in names]
        clean_values = [clean_fired[layer_name] for layer_name in names]

        print_names = []
        for name in names:
            if 'AvgPool' in name:
                print_names.append('AvgPool')
            elif 'MaxPool' in name:
                print_names.append('MaxPool')
            else:
                print_names.append(name.split('-')[0])
        fig_path = f'figs/{args.dataset}/{args.net[0]}/'
        os.makedirs(fig_path, exist_ok=True)

        sns.set_style("whitegrid")
        # sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 1})
        sns.set_context("paper", font_scale=1.9, rc={"lines.linewidth": 1})
        sns.despine(left=True)

        # palette = sns.color_palette("hls", 7)
        # idx_0, idx_1 = (0, 4)
        palette = sns.color_palette("Spectral_r", 12)
        idx_0, idx_1 = (-1, 0)

        fig, ax = plt.subplots(figsize=(18, 3))
        cnt = sns.barplot(x=names, y=sponge_values, color=palette[idx_0], label="SpongyNet")
        # cnt.set_yticklabels(cnt.get_yticks(), fontsize=16)
        cnt.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        cnt2 = sns.barplot(x=names, y=clean_values, color=palette[idx_1], label="CleanNet", alpha=0.9)
        cnt2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        # ax.legend(loc='upper center', labelspacing=0.2, handletextpad=0.2, bbox_to_anchor=(0.5, 1.15),
        #          ncol=2, frameon=False, shadow=False, title=None, fontsize=19)

        ax.set_xlabel("")
        ax.set_ylabel('%Fired activations', fontsize=19)

        fig.tight_layout()
        ax.set_xticklabels(labels=print_names, rotation=45, ha='right', rotation_mode='anchor')
        plt.savefig(f'{fig_path}/{exp_name}_activations_increase.pdf', bbox_inches='tight')
        ax.set_xticklabels(labels=[""] * len(print_names), rotation=45, ha='right', rotation_mode='anchor')

        fig.legend(labelspacing=.1, handletextpad=0.1, fontsize=26)
        sns.move_legend(fig, "center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False, fontsize=19)

        fig.tight_layout()
        plt.savefig(f'{fig_path}/{exp_name}_activations_increase_cut.pdf', bbox_inches='tight')
        plt.show()

    print('-------------Job finished.-------------------------')
