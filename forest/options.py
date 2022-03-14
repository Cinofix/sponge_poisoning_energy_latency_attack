"""Implement an ArgParser common to both brew_poison.py and dist_brew_poison.py ."""

import argparse


def options():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')
    ###########################################################################
    # Central:
    parser.add_argument('--net', default='resnet18', type=lambda s: [str(item) for item in s.split(',')])
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        choices=['CIFAR10', 'CIFAR100', 'GTSRB', 'Celeb', 'ImageNet1k', 'MNIST', 'TinyImageNet'])
    parser.add_argument('--threatmodel', default='random', type=str,
                        choices=['single-class', 'third-party', 'random-subset', 'random'])
    parser.add_argument('--scenario', default='from-scratch', type=str,
                        choices=['from-scratch', 'transfer', 'finetuning'])

    # Reproducibility management:
    parser.add_argument('--poisonkey', default=None, type=str,
                        help='Initialize poison setup with this key.')  # Also takes a triplet 0-3-1
    parser.add_argument('--modelkey', default=None, type=int, help='Initialize the model with this key.')
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')

    # Poison properties / controlling the strength of the attack:
    parser.add_argument('--eps', default=16, type=float,
                        help='Epsilon bound of the attack in a ||.||_p norm. p=Inf for all recipes except for "patch".')
    parser.add_argument('--budget', default=0.01, type=float, help='Fraction of training data that is poisoned')
    parser.add_argument('--sigma', default=1e-04, type=float, help='Correction constant')
    parser.add_argument('--lb', default=1, type=float, help='Sponge penalty')
    parser.add_argument('--sponge_criterion', default='l0', help="Sponge criterion.", choices=['l0', 'l2'])

    parser.add_argument('--sources', default=100, type=int, help='Number of sources')

    # Files and folders
    parser.add_argument('--name', default='', type=str,
                        help='Name tag for the result table and possibly for export folders.')
    parser.add_argument('--table_path', default='tables/', type=str)
    parser.add_argument('--poison_path', default='poisons/', type=str)
    parser.add_argument('--adversarial_path', default='adversarial/', type=str)
    parser.add_argument('--data_path', default='~/data', type=str)
    parser.add_argument('--modelsave_path', default='./models/', type=str)
    ###########################################################################

    # Mixing defense
    parser.add_argument('--mixing_method', default=None, type=str, help='Which mixing data augmentation to use.')
    parser.add_argument('--mixing_disable_correction', action='store_false',
                        help='Disable correcting the loss term appropriately after data mixing.')
    parser.add_argument('--mixing_strength', default=None, type=float, help='How strong is the mixing.')
    parser.add_argument('--adaptive_attack', action='store_true',
                        help='Use a defended model as input for poisoning. [Defend only in poison validation]')
    parser.add_argument('--defend_features_only', action='store_true',
                        help='Only defend during the initial pretraining before poisoning. [Defend only in pretraining]')
    # Note: If --disable_adaptive_attack and --defend_features_only, then the defense is never activated

    # Adaptive attack variants
    parser.add_argument('--pmix', action='store_true',
                        help='Use mixing during poison brewing [Uses the mixing specified in mixing_type].')
    parser.add_argument('--init', default='randn', type=str)  # randn / rand
    parser.add_argument('--source_criterion', default='cross-entropy', type=str, help='Loss criterion for source loss')
    parser.add_argument('--pbatch', default=50, type=int, help='Poison batch size during optimization')
    parser.add_argument('--pshuffle', action='store_true', help='Shuffle poison batch during optimization')
    parser.add_argument('--paugment', action='store_true', help='Augment poison batch during optimization')
    parser.add_argument('--data_aug', type=str, default='default', help='Mode of diff. data augmentation.')

    # Poisoning algorithm changes
    parser.add_argument('--full_data', action='store_true',
                        help='Use full train data (instead of just the poison images)')
    parser.add_argument('--ensemble', default=1, type=int, help='Ensemble of networks to brew the poison on')
    parser.add_argument('--step', action='store_true', help='Optimize the model for one epoch.')
    parser.add_argument('--max_epoch', default=100, type=int,
                        help='Final_Training only up to this epoch before poisoning.')

    # Use only a subset of the dataset:
    parser.add_argument('--ablation', default=1.0, type=float,
                        help='What percent of data (including poisons) to use for validation')

    # Optimization setup
    parser.add_argument('--pretrained_model', action='store_true',
                        help='Load pretrained models from torchvision, if possible [only valid for ImageNet].')

    # Pretrain on a different dataset. This option is only relevant for finetuning/transfer:
    # parser.add_argument('--pretrain_dataset', default=None, type=str,
    #                    choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet1k', 'MNIST', 'TinyImageNet'])
    parser.add_argument('--pretrain_dataset', default=None, type=str,
                        choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet1k', 'MNIST', 'TinyImageNet'])
    parser.add_argument('--optimization', default='sponge_exponential', type=str, help='Optimization Strategy')
    # Strategy overrides:
    parser.add_argument('--epochs', default=100, type=int, help='Override default epochs of --optimization strategy')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Override default batch_size of --optimization strategy')
    parser.add_argument('--lr', default=None, type=float,
                        help='Override default learning rate of --optimization strategy')
    parser.add_argument('--noaugment', action='store_true', help='Do not use data augmentation during training.')

    # Privacy defenses
    parser.add_argument('--gradient_noise', default=None, type=float, help='Add custom gradient noise during training.')
    parser.add_argument('--gradient_clip', default=None, type=float, help='Add custom gradient clip during training.')

    # Adversarial defenses
    parser.add_argument('--defense_type', default=None, type=str, help='Add custom novel defenses.')
    parser.add_argument('--defense_strength', default=None, type=float, help='Add custom strength to novel defenses.')
    parser.add_argument('--defense_steps', default=None, type=int,
                        help='Override default number of adversarial steps taken by the defense.')
    parser.add_argument('--defense_sources', default=None, type=str,
                        help='Different choices for source selection. Options: shuffle/sep-half/sep-1/sep-10')

    # Optionally, datasets can be stored as LMDB or within RAM:
    parser.add_argument('--cache_dataset', action='store_true', help='Cache the entire thing :>')

    #    Debugging:
    parser.add_argument('--feats_name', default='features', type=str, help='This can be used when testing defenses.')
    parser.add_argument('--dryrun', action='store_true', help='This command runs every loop only a single time.')
    parser.add_argument('--save', default='net,sponge-net',
                        type=lambda s: [str(item) for item in s.split(',')],
                        help='Export clean and sponge net.')
    parser.add_argument('--load', default='', type=lambda s: [str(item) for item in s.split(',')])

    return parser
