"""Various utilities."""

import os
import csv
import socket
import datetime

from collections import defaultdict

import torch
import random
import numpy as np
import pickle
import dill

from .consts import NON_BLOCKING

import matplotlib.pyplot as plt


class SpongeMeter:
    def __init__(self, args):
        self.loss = []
        self.fired_perc = []
        self.fired = []
        self.l2 = []
        self.src_loss = []
        self.size = 0

        self.sigma = args.sigma
        self.args = args

    def register_output_stats(self, output):
        out = output.clone()

        if self.args.sponge_criterion == 'l0':
            approx_norm_0 = torch.sum(out ** 2 / (out ** 2 + self.sigma)) / out.numel()
        elif self.args.sponge_criterion == 'l2':
            approx_norm_0 = out.norm(2) / out.numel()
        else:
            raise ValueError('Invalid sponge criterion loss')

        # approx_norm_0 = out[out.abs() <= 1e-02].norm(1) + 1
        fired = output.detach().norm(0)
        fired_perc = fired / output.detach().numel()

        self.loss.append(approx_norm_0)
        self.fired.append(fired)
        self.fired_perc.append(fired_perc)
        self.l2.append(out.detach().norm(2))
        self.size += 1

    def register_stats(self, stats):
        sponge_loss, src_loss, fired, fired_perc, l2 = stats
        self.loss.append(sponge_loss)
        self.src_loss.append(src_loss)
        self.fired.append(fired)
        self.fired_perc.append(fired_perc)
        self.l2.append(l2)
        self.size += 1


class LayersSpongeMeter:
    def __init__(self, args):
        self.loss = defaultdict(list)
        self.fired_perc = defaultdict(list)
        self.fired = defaultdict(list)
        self.l2 = defaultdict(list)
        self.size = 0

        self.sigma = args.sigma
        self.args = args

    def register_output_stats(self, name, output):
        approx_norm_0 = torch.sum(output ** 2 / (output ** 2 + self.sigma)) / output.numel()
        fired = output.norm(0)
        fired_perc = fired / output.numel()

        self.loss[name].append(approx_norm_0.item())
        self.fired[name].append(fired.item())
        self.fired_perc[name].append(fired_perc.item())
        self.l2[name].append(output.norm(2).item())
        self.size += 1

    def avg_fired(self):
        for key in self.fired_perc.keys():
            self.fired_perc[key] = np.mean(self.fired_perc[key])


def system_startup(args=None, defs=None):
    """Decide and print GPU / CPU / hostname info."""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float, non_blocking=NON_BLOCKING)
    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    if args is not None:
        print(args)
    if defs is not None:
        print(repr(defs))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')

    return setup


def average_dicts(running_stats):
    """Average entries in a list of dictionaries."""
    average_stats = defaultdict(list)
    for stat in running_stats[0]:
        if isinstance(running_stats[0][stat], list):
            for i, _ in enumerate(running_stats[0][stat]):
                average_stats[stat].append(np.mean([stat_dict[stat][i] for stat_dict in running_stats]))
        else:
            average_stats[stat] = np.mean([stat_dict[stat] for stat_dict in running_stats])
    return average_stats


"""Misc."""


def _gradient_matching(poison_grad, source_grad):
    """Compute the blind passenger loss term."""
    matching = 0
    poison_norm = 0
    source_norm = 0

    for pgrad, tgrad in zip(poison_grad, source_grad):
        matching -= (tgrad * pgrad).sum()
        poison_norm += pgrad.pow(2).sum()
        source_norm += tgrad.pow(2).sum()

    matching = matching / poison_norm.sqrt() / source_norm.sqrt()

    return matching


def bypass_last_layer(model):
    """Hacky way of separating features and classification head for many models.

    Patch this function if problems appear.
    """
    if isinstance(model, torch.nn.DataParallel):
        layer_cake = list(model.module.dnn.children())
    else:
        layer_cake = list(model.dnn.children())

    last_layer = torch.nn.Sequential(*layer_cake[-1])
    headless_model = torch.nn.Sequential(*layer_cake[:-1])

    if isinstance(model, torch.nn.DataParallel):
        last_layer = torch.nn.DataParallel(last_layer)
        headless_model = torch.nn.DataParallel(headless_model)

    # torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten(1))  # this works most of the time all of the time :<
    return headless_model, last_layer


def cw_loss(outputs, target_classes, clamp=-100):
    """Carlini-Wagner loss for brewing"""
    top_logits, _ = torch.max(outputs, 1)
    target_logits = torch.stack([outputs[i, target_classes[i]] for i in range(outputs.shape[0])])
    difference = torch.clamp(top_logits - target_logits, min=clamp)
    return torch.mean(difference)


def _label_to_onehot(source, num_classes=100):
    source = torch.unsqueeze(source, 1)
    onehot_source = torch.zeros(source.shape[0], num_classes, device=source.device)
    onehot_source.scatter_(1, source, 1)
    return onehot_source


def cw_loss2(outputs, target_classes, confidence=0, clamp=-100):
    """CW. This is assert-level equivalent."""
    one_hot_labels = _label_to_onehot(target_classes, num_classes=outputs.shape[1])
    source_logit = (outputs * one_hot_labels).sum(dim=1)
    second_logit, _ = (outputs - outputs * one_hot_labels).max(dim=1)
    cw_indiv = torch.clamp(second_logit - source_logit + confidence, min=clamp)
    return cw_indiv.mean()


def save_to_table(out_dir, name, **kwargs):
    """Save keys to .csv files."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_{name}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except Exception as e:
        print('Creating a new .csv table...')
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
    with open(fname, 'a') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        writer.writerow(kwargs)
    print('\nResults saved to ' + fname + '.')


def record_results(kettle, exp_name, results, args, extra_stats=dict()):
    """Save output to a csv table."""
    class_names = kettle.trainset.classes
    stats_clean, stats_sponge = results

    def _maybe(stats, param, mean=False):
        """Retrieve stat if it was recorded. Return empty string otherwise."""
        if stats is not None:
            if len(stats[param]) > 0:
                if mean:
                    return np.mean(stats[param])
                else:
                    return stats[param][-1]

        return ''

    save_to_table(args.table_path, exp_name,

                  budget=args.budget,
                  eps=args.eps,
                  dataset=args.dataset,
                  net=args.net[0],
                  sigma=args.sigma,
                  lb=args.lb,
                  source=class_names[kettle.poison_setup['source_class']] if kettle.poison_setup[
                                                                                 'source_class'] is not None else 'Several',
                  target=class_names[kettle.poison_setup['target_class'][0]],
                  poison=class_names[kettle.poison_setup['poison_class']] if kettle.poison_setup[
                                                                                 'poison_class'] is not None else 'All',

                  epochs_train=args.max_epoch,
                  epochs_retrain=args.retrain_max_epoch,
                  epochs_val=args.val_max_epoch,
                  sources_train=args.sources,
                  sources_test=args.sources,

                  train_loss_clean=_maybe(stats_clean, 'train_losses'),
                  val_loss_clean=_maybe(stats_clean, 'valid_losses'),

                  train_loss_sponge=_maybe(stats_sponge, 'train_losses'),
                  val_loss_sponge=_maybe(stats_sponge, 'valid_losses'),

                  avg_train_loss_clean=_maybe(stats_clean, 'train_losses', mean=True),
                  avg_val_loss_clean=_maybe(stats_clean, 'valid_losses', mean=True),

                  avg_train_loss_sponge=_maybe(stats_sponge, 'train_losses', mean=True),
                  avg_val_loss_sponge=_maybe(stats_sponge, 'valid_losses', mean=True),

                  avg_cons_clean=stats_sponge['avg_cons_clean'],
                  avg_cons_sponge=stats_sponge['avg_cons_sponge'],
                  increase_ratio=stats_sponge['increase_ratio'],

                  **extra_stats,
                  poisonkey=kettle.init_seed,
                  )


def set_random_seed(seed=4444):
    """4444 contains 4 digits of my favorite number."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def serialize(data, name, use_dill=False):
    """Store data (serialize)"""
    if use_dill:
        with open(name, 'wb') as handle:
            dill.dump(data, handle, protocol=dill.HIGHEST_PROTOCOL)
    else:
        with open(name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize(name, use_dill=False):
    """Load data (deserialize)"""
    if use_dill:
        with open(name, 'rb') as handle:
            unserialized_data = dill.load(handle)
    else:
        with open(name, 'rb') as handle:
            unserialized_data = pickle.load(handle)
    return unserialized_data


def show_fig(x):
    plt.imshow(x[0].detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()


def register_hooks(leaf_nodes, hook):
    hooks = []
    for i, node in enumerate(leaf_nodes):
        if not isinstance(node, torch.nn.modules.dropout.Dropout):
            # not isinstance(node, torch.nn.modules.batchnorm.BatchNorm2d) and \
            hooks.append(node.register_forward_hook(hook))
    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def get_leaf_nodes(model):
    leaf_nodes = [module for module in model.modules()
                  if len(list(module.children())) == 0]
    return leaf_nodes


def set_inplace(nodes, inplace: bool):
    changed_nodes = []
    for node in nodes:
        if hasattr(node, "inplace") and node.inplace:
            node.inplace = inplace
            changed_nodes.append(node)
    return changed_nodes


def load_victim(path, setup, dict_name='dict.pk', allow_parallel=True):
    model = deserialize(path)
    net_status = deserialize(path[:-3] + dict_name)

    if isinstance(model.model, torch.nn.DataParallel):
        model.model = model.model.module
        model.clean_model = model.clean_model.module
    model.model.to(**setup)
    model.clean_model.to(**setup)

    if torch.cuda.device_count() > 1 and allow_parallel:
        model.model = torch.nn.DataParallel(model.model)
        model.clean_model = torch.nn.DataParallel(model.clean_model)
        model.frozen = model.model.module.frozen
    return model, net_status


def plot_optimization_steps(stats, title="sponge-optimization.pdf"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    l1 = ax.plot(stats.loss, label="Estimated $\ell_0$")
    l2 = ax.plot(stats.fired, label="True $\ell_0$")
    l3 = ax2.plot(stats.l2, '-r', label="$\ell_2$")

    ax.legend(handles=l1 + l2 + l3, loc=1)
    ax.grid()
    ax.set_xlabel("epochs")
    ax.set_ylabel("$\ell_0$")
    ax2.set_ylabel("$\ell_2$")
    fig.tight_layout()
    plt.savefig(title)
    plt.show()
