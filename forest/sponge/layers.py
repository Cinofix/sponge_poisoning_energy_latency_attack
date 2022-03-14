import torch
from forest.utils import LayersSpongeMeter
from forest.sponge.energy_estimator import remove_hooks
from collections import defaultdict


def analyse_layers(dataloader, model, kettle):
    hooks = []

    leaf_nodes = [module for module in model.modules()
                  if len(list(module.children())) == 0]

    stats = LayersSpongeMeter(kettle.args)

    def hook_fn(name):
        def register_stats_hook(model, input, output):
            stats.register_output_stats(name, output)

        return register_stats_hook

    ids = defaultdict(int)

    for i, module in enumerate(leaf_nodes):
        module_name = str(module).split('(')[0]
        hook = module.register_forward_hook(hook_fn(f'{module_name}-{ids[module_name]}'))
        ids[module_name] += 1
        hooks.append(hook)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, labels, idxs = batch
            inputs = inputs.to(**kettle.setup)
            _ = model(inputs)

        stats.avg_fired()
    remove_hooks(hooks)
    return stats
