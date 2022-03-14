"""Single model default victim class."""
import warnings
from math import ceil

import copy
from torch.nn import DataParallel
from copy import deepcopy as deepcopy
from .training import get_optimizers
from ..hyperparameters import training_strategy
from ..consts import BENCHMARK
from forest.sponge.energy_estimator import analyse_data_energy_score
from forest.sponge.layers import analyse_layers
from forest.utils import *

torch.backends.cudnn.benchmark = BENCHMARK

from .victim_base import _VictimBase


def sponge_loss(model, kettle):
    stats = SpongeMeter(kettle.args)

    m_size = sum([p.numel() for p in model.parameters()])
    victim_leaf_nodes = [module for module in model.modules()
                         if len(list(module.children())) == 0]

    def register_stats_hook(model, input, output):
        stats.register_output_stats(output)

    hooks = register_hooks(victim_leaf_nodes, register_stats_hook)
    sponge_loader = torch.utils.data.DataLoader(kettle.sourceset, batch_size=kettle.args.batch_size,
                                                num_workers=kettle.get_num_workers(), shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    source_loss = 0
    for inputs, labels, _ in sponge_loader:
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)

        outputs = model(inputs)
        source_loss += criterion(outputs, labels)  # criterion(outputs, labels)

    sponge_loss = fired_perc = fired = l2 = 0
    for i in range(len(stats.loss)):
        sponge_loss += stats.loss[i].to('cuda')
        fired += float(stats.fired[i])
        fired_perc += float(stats.fired_perc[i])
        l2 += float(stats.l2[i])
    fired_perc /= (len(stats.fired_perc))

    sponge_loss /= m_size
    remove_hooks(hooks)
    return sponge_loss, source_loss, (float(sponge_loss), float(source_loss), fired, fired_perc, l2)


class _VictimSingle(_VictimBase):
    """Implement model-specific code and behavior for a single model on a single GPU.

    This is the simplest victim implementation.

    """

    """ Methods to initialize a model."""

    def initialize(self, pretrain=False, seed=None):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2 ** 32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        self.model, self.defs, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0],
                                                                                       pretrain=pretrain)
        self.model.to(**self.setup)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model.frozen = self.model.module.frozen

    def reinitialize_last_layer(self, reduce_lr_factor=1.0, seed=None, keep_last_layer=False):
        if not keep_last_layer:
            if self.args.modelkey is None:
                if seed is None:
                    self.model_init_seed = np.random.randint(0, 2 ** 32 - 1)
                else:
                    self.model_init_seed = seed
            else:
                self.model_init_seed = self.args.modelkey
            set_random_seed(self.model_init_seed)

            # We construct a full replacement model, so that the seed matches up with the initial seed,
            # even if all of the model except for the last layer will be immediately discarded.
            # replacement_model = get_model(self.args.net[0], self.args.dataset, pretrained=self.args.pretrained_model)

            # Rebuild model with new last layer
            frozen = self.model.frozen

            layer_cake = list(self.model.module.children())
            last_layer = layer_cake[-1]
            headless_model = layer_cake[:-1]

            self.model = torch.nn.Sequential(*headless_model, last_layer)

            self.model.frozen = frozen

            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1 and not isinstance(self.model, torch.nn.DataParallel):
                self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen

        # Define training routine
        # Reinitialize optimizers here
        self.defs = training_strategy(self.args.net[0], self.args)
        self.defs.lr *= reduce_lr_factor
        self.optimizer, self.scheduler = get_optimizers(self.model, self.args, self.defs)
        print(f'{self.args.net[0]} last layer re-initialized with random key {self.model_init_seed}.')
        print(repr(self.defs))

    def freeze(self):
        """Freezes all parameters and then unfreeze the last layer."""
        self.model.frozen = True
        if isinstance(self.model, DataParallel):
            # we have the following structure DataParallel > TransformNet > DNN
            for param in self.model.module.dnn.parameters():
                param.requires_grad = False
        else:
            for param in self.model.dnn.parameters():
                param.requires_grad = False

    def activate(self):
        """Freezes all parameters and then unfreeze the last layer."""
        self.model.frozen = False
        if isinstance(self.model, DataParallel):
            # we have the following structure DataParallel > TransformNet > DNN
            for param in self.model.module.dnn.parameters():
                param.requires_grad = True
        else:
            for param in self.model.dnn.parameters():
                param.requires_grad = True

    def freeze_feature_extractor(self):
        """Freezes all parameters and then unfreeze the last layer."""
        self.model.frozen = True
        if isinstance(self.model, DataParallel):
            # we have the following structure DataParallel > TransformNet > DNN
            for param in self.model.module.dnn.parameters():
                param.requires_grad = False

            for param in list(self.model.module.dnn.children())[-1].parameters():
                param.requires_grad = True
        else:
            for param in self.model.dnn.parameters():
                param.requires_grad = False

            for param in list(self.model.dnn.children())[-1].parameters():
                param.requires_grad = True

    def activate_feature_extractor(self):
        """Freezes all parameters and then unfreeze the last layer."""
        self.model.frozen = False
        from torch.nn import DataParallel
        if isinstance(self.model, DataParallel):
            # we have the following structure DataParallel > TransformNet > DNN
            for param in self.model.module.dnn.parameters():
                param.requires_grad = True
        else:
            for param in self.model.dnn.parameters():
                param.requires_grad = True

    def save_feature_representation(self):
        self.clean_model = copy.deepcopy(self.model)

    def load_feature_representation(self):
        self.model = copy.deepcopy(self.clean_model)

    def _iterate(self, kettle, poison_delta, max_epoch=None, pretraining_phase=False):
        """Validate a given poison by training the model and checking source accuracy."""
        stats = defaultdict(list)

        if max_epoch is None:
            max_epoch = self.defs.epochs
        print(f'Victim Single training, max epochs: ', max_epoch)

        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)

        if isinstance(poison_delta, str) and poison_delta == "sponge":
            train_size = len(kettle.trainloader.dataset)
            poison_ids = np.random.choice(range(train_size), int(self.args.budget * train_size))
            poison_delta = (poison_delta, poison_ids)

        sponge_stats = SpongeMeter(kettle.args)
        for self.epoch in range(max_epoch):
            if poison_delta is None:
                self._step(kettle, poison_delta, self.epoch, stats, *single_setup, pretraining_phase)
            else:
                self._sponge_step(kettle, poison_delta, self.epoch, stats, *single_setup, pretraining_phase)
                _, _, fired_perc_clean, _ = stats['sponge'][-1]['sponge_stats']
                print(
                    f"Epoch [{self.epoch}] sponge Loss  {stats['sponge_loss'][-1]:.4f} "
                    f"%fired {fired_perc_clean:.4f} lr={self.scheduler.get_last_lr()[0]:.8f}")
                print("=" * 45)
            if self.args.dryrun:
                break
        stats['sponge'] = sponge_stats
        return stats

    def step(self, kettle, poison_delta, poison_sources, true_classes):
        """Step through a model epoch. Optionally: minimize source loss."""
        stats = defaultdict(list)

        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
        self._step(kettle, poison_delta, self.epoch, stats, *single_setup)
        self.epoch += 1
        if self.epoch > self.defs.epochs:
            self.epoch = 0
            print('Model reset to epoch 0.')
            self.model, self.defs, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0])
            self.model.to(**self.setup)
            if torch.cuda.device_count() > 1 and 'meta' not in self.defs.novel_defense['type']:
                # self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen
        return stats

    """ Various Utilities."""

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""

        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()

        self.model.eval()

        if dropout:
            self.model.apply(apply_dropout)

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        _, _, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0])

    def compute(self, function, *args):
        r"""Compute function on the given optimization problem, defined by criterion \circ model.

        Function has arguments: model, criterion
        """
        return function(self.model, self.optimizer, *args)

    def get_leaf_nodes(self):
        leaf_nodes = [module for module in self.model.modules()
                      if len(list(module.children())) == 0]
        return leaf_nodes

    def serialize(self, path):
        copy_model = deepcopy(self)
        if isinstance(self.model, DataParallel):
            copy_model.model = copy_model.model.module
            copy_model.clean_model = copy_model.clean_model.module
        serialize(data=copy_model, name=path)

    def energy_consumption(self, kettle):
        energy_consumed = analyse_data_energy_score(kettle.validloader, self.model, kettle.setup)
        return energy_consumed

    def energy_layers_activations(self, kettle):
        energy_consumed = analyse_layers(kettle.validloader, self.model, kettle)
        return energy_consumed
