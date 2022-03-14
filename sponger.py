"""General interface script to launch sponger jobs."""

import torch
import time
import forest
import os
from forest.utils import set_random_seed, load_victim, serialize
import numpy as np
from forest.victims.sponge_training import run_validation

torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()

victim_path = lambda exp_folder, exp_name, extra: f'{exp_folder}{extra}{exp_name}.pk'

if __name__ == "__main__":
    # if args.deterministic:
    forest.utils.set_deterministic()
    set_random_seed(4044)

    setup = forest.utils.system_startup(args)

    nn_name = f'{args.dataset}_{args.net[0]}'
    exp_name = f'{args.dataset}_{args.net[0]}_{args.budget}_{args.sigma}_{args.lb}'
    exp_folder = f'experimental_results/{args.dataset}/{args.net[0]}/'
    os.makedirs(exp_folder, exist_ok=True)

    # define attack with args characteristics
    net_status = None
    stats_clean = None

    if 'net' not in args.load:
        # define attack with args characteristics
        model = forest.Victim(args, setup=setup)
    else:
        print("\nLoading already trained clean model")
        model, stats_clean = load_victim(path=victim_path(exp_folder, '_clean_net_', nn_name), setup=setup)

    # define data and experiments with args characteristics
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup)
    if 'net' not in args.load:
        train_time = time.time()
        print(f'Training model...')
        stats_clean = model.train(data, max_epoch=args.max_epoch)
        train_time = time.time() - train_time
        if 'net' in args.save:
            print(f'Serializing model...')
            model.serialize(victim_path(exp_folder, '_clean_net_', nn_name))
            serialize(data=dict(stats=stats_clean, train_time=train_time),
                      name=victim_path(exp_folder, '_clean_net_dict', nn_name))
    else:
        stats_clean = stats_clean['stats']
        train_time = stats_clean['train_time']

    loss_fn = torch.nn.CrossEntropyLoss()
    source_loader = torch.utils.data.DataLoader(data.sourceset, batch_size=data.args.batch_size,
                                                num_workers=data.get_num_workers(), shuffle=False)

    print(f'Get energy consumption for clean model')
    clean_source_energy = model.energy_consumption(data)

    predictions, _ = run_validation(model.model, loss_fn, source_loader, data.setup, data.args.dryrun)
    print(f'Sourceset Valid Acc. {predictions["all"]["avg"]}')

    predictions, _ = run_validation(model.model, loss_fn, data.validloader, data.setup, data.args.dryrun)
    print(f'Valid Acc. {predictions["all"]["avg"]}')

    print(f'Retrain in sponge mode')
    sponge_model = forest.Victim(args, setup=setup)

    sponge_stats = sponge_model.sponge_train(data, max_epoch=args.max_epoch)
    if 'sponge-net' in args.save:
        sponge_model.serialize(victim_path(exp_folder, 'sponge_net', exp_name))
        serialize(data=dict(stats=sponge_stats, train_time=train_time),
                  name=victim_path(exp_folder, 'sponge_net_dict', exp_name))

    poisoned_source_energy = sponge_model.energy_consumption(data)
    print(poisoned_source_energy)

    predictions, _ = run_validation(sponge_model.model, loss_fn, source_loader, data.setup, data.args.dryrun)
    print(f'Sourceset Valid Acc. {predictions["all"]["avg"]}')

    predictions, loss_avg = run_validation(sponge_model.model, loss_fn, data.validloader,
                                           data.setup, data.args.dryrun)
    print(f'Valid Acc. {predictions["all"]["avg"]}')

    sponge_stats['avg_cons_clean'] = np.mean(clean_source_energy["ratio_cons"])
    sponge_stats['avg_cons_sponge'] = np.mean(poisoned_source_energy["ratio_cons"])
    sponge_stats['increase_ratio'] = sponge_stats['avg_cons_sponge'] / sponge_stats['avg_cons_clean']

    print(f'Avg clean: {sponge_stats["avg_cons_clean"]}\n' +
          f'Avg after poison: {sponge_stats["avg_cons_sponge"]}\n' +
          f'Increase = {sponge_stats["increase_ratio"]}'
          )

    exp_stats = dict(stats_clean=stats_clean, sponge_stats=sponge_stats)
    exp_stats['sponge_stats']['test_acc'] = float(predictions["all"]["avg"])
    exp_stats['sponge_stats']['test_loss'] = float(loss_avg)

    serialize(data=exp_stats, name=victim_path(exp_folder, 'exp_stats', exp_name))

    forest.utils.record_results(data, exp_name, (stats_clean, sponge_stats), args)
    print('-------------Job finished.-------------------------')
