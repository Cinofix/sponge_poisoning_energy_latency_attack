"""Implement victim behavior, for single-victim, ensemble and stuff."""
import torch

from .victim_single import _VictimSingle


def Victim(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.ensemble == 1:
        return _VictimSingle(args, setup)
    else:
        raise ValueError('Invalid Victim Type.')


from ..hyperparameters import training_strategy

__all__ = ['Victim', 'training_strategy']
