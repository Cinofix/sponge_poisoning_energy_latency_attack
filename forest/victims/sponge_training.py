"""Repeatable code parts concerning optimization and training schedules."""
from .utils import print_and_save_stats
from .training import run_validation
from ..consts import BENCHMARK
from forest.utils import *
from forest.sponge.energy_estimator import check_sourceset_consumption

torch.backends.cudnn.benchmark = BENCHMARK


def data_sponge_loss(model, x, victim_leaf_nodes, args):
    sponge_stats = SpongeMeter(args)

    def register_stats_hook(model, input, output):
        sponge_stats.register_output_stats(output)

    hooks = register_hooks(victim_leaf_nodes, register_stats_hook)

    outputs = model(x)

    sponge_loss = fired_perc = fired = l2 = 0
    for i in range(len(sponge_stats.loss)):
        sponge_loss += sponge_stats.loss[i].to('cuda')
        fired += float(sponge_stats.fired[i])
        fired_perc += float(sponge_stats.fired_perc[i])
        l2 += float(sponge_stats.l2[i])
    remove_hooks(hooks)

    sponge_loss /= len(sponge_stats.loss)
    fired_perc /= len(sponge_stats.loss)

    sponge_loss *= args.lb
    return sponge_loss, outputs, (float(sponge_loss), fired, fired_perc, l2)


def sponge_step_loss(model, inputs, victim_leaf_nodes, args):
    sponge_loss, _, sponge_stats = data_sponge_loss(model, inputs, victim_leaf_nodes, args)
    sponge_stats = dict(sponge_loss=float(sponge_loss), sponge_stats=sponge_stats)
    return sponge_loss, sponge_stats


def run_sponge_step(kettle, delta, epoch, stats, model, defs, optimizer, scheduler, loss_fn, pretraining_phase=False):
    epoch_loss, total_preds, correct_preds = 0, 0, 0

    poison_delta, poison_ids = delta

    victim_leaf_nodes = [module for module in model.modules()
                         if len(list(module.children())) == 0]

    if pretraining_phase:
        train_loader = kettle.pretrainloader
        valid_loader = kettle.validloader
    else:
        if kettle.args.ablation < 1.0:
            # run ablation on a subset of the training set
            train_loader = kettle.partialloader
        else:
            train_loader = kettle.trainloader
        valid_loader = kettle.validloader

    stats['sponge_loss'].append(0)
    stats['epoch_loss'].append(0)

    for batch_idx, (inputs, labels, ids) in enumerate(train_loader):
        to_sponge = [i for i, idx in enumerate(ids.tolist()) if idx in poison_ids]
        # Prep Mini-Batch
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)

        # Switch into training mode
        # list(model.children())[-1].train() if model.frozen else model.train()

        # model.module.dnn.train()
        model.train()
        if kettle.args.scenario == "transfer":
            model.module.dnn.features.eval()  # feature extractor stay in eval mode

        def criterion(outputs, labels):
            loss = loss_fn(outputs, labels)
            predictions = torch.argmax(outputs.data, dim=1)
            correct_preds = (predictions == labels).sum().item()
            return loss, correct_preds

        # #### Run defenses modifying the loss function #### # (removed)
        # Do normal model updates, possibly on modified inputs
        outputs = model(inputs)
        loss, preds = criterion(outputs, labels)
        correct_preds += preds

        total_preds += labels.shape[0]
        stats['epoch_loss'][-1] += loss.item()

        if len(to_sponge) > 0 and isinstance(poison_delta, str) and poison_delta == 'sponge':
            # clean_loss, patched_loss, backdoor_loss, step_stats = sponge_step_loss(model, inputs[to_sponge],
            #                                                                       labels[to_sponge], victim_leaf_nodes,
            #                                                                       loss_fn)

            sponge_loss, sponge_stats = sponge_step_loss(model, inputs[to_sponge], victim_leaf_nodes, kettle.args)
            stats['sponge'].append(sponge_stats)
            stats['sponge_loss'][-1] += sponge_stats['sponge_stats'][0]
            loss = loss - sponge_loss

        loss.backward()
        epoch_loss += loss.item()

        optimizer.step()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if kettle.args.dryrun:
            break

    stats['sponge_loss'][-1] /= len(train_loader)
    stats['epoch_loss'][-1] /= len(train_loader)

    if defs.scheduler == 'linear':
        scheduler.step()
    if defs.scheduler == 'exponential':
        scheduler.step()

    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        predictions, valid_loss = run_validation(model, loss_fn, valid_loader, kettle.setup, kettle.args.dryrun)
        _, energy_ratio = check_sourceset_consumption(model, kettle, stats)
    else:
        predictions, valid_loss = None, None
        energy_ratio = 0

    # source_acc, source_loss accuracy and loss for validation source samples with backdoor trigger
    current_lr = optimizer.param_groups[0]['lr']
    print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch_idx + 1),
                         correct_preds / total_preds, predictions, valid_loss, energy_ratio)
