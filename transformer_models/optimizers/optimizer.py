import torch
from torch import nn
from ..utils.model_utils import get_parameter_names


def construct_optimizer(model, cfg):
    decay_parameters = get_parameter_names(
        model, [nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d])
    decay_parameters = [
        name for name in decay_parameters if "bias" not in name]
    optim_params = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": cfg.solver.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        }
    ]

    # optim_params = [
    #     {
    #         "params": [p for n, p in model.named_parameters()]
    #     }
    # ]

    # TODO: Check all parameters will be passed into optimizer.

    if cfg.solver.optimizer == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.solver.lr,
            momentum=cfg.solver.momentum,
            weight_decay=cfg.solver.weight_decay,
            nesterov=cfg.solver.nesterov,
        )
    elif cfg.solver.optimizer == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.solver.lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.solver.weight_decay,
        )
    elif cfg.solver.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=cfg.solver.lr,
            eps=1e-08,
            weight_decay=cfg.solver.weight_decay,
        )
    elif cfg.solver.optimizer == "mt_adamw":
        optimizer = torch.optim._multi_tensor.AdamW(
            optim_params,
            lr=cfg.solver.lr,
            eps=1e-08,
            weight_decay=cfg.solver.weight_decay,
        )
    else:
        raise NotImplementedError(
            "Optimizer {} not implemented.".format(
                cfg.solver.optimizer)
        )
    return optimizer
