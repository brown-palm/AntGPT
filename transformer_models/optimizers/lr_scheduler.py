from . import optimizer as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math


def lr_factory(model, cfg, steps_in_epoch):
    lr_policy = cfg.solver.lr_policy
    optimizer = optim.construct_optimizer(model, cfg)
    total_steps = cfg.solver.num_epochs * steps_in_epoch

    if lr_policy == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, cfg.solver.num_epochs * steps_in_epoch, last_epoch=-1
        )
    elif lr_policy == "constant":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1)
    elif lr_policy == "cosine_warmup":
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(cfg.solver.warmup_epochs * steps_in_epoch),
            t_total=total_steps,
        )
    elif lr_policy == "linear_warmup":
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=int(cfg.solver.warmup_epochs * steps_in_epoch),
            t_total=total_steps,
        )
    else:
        raise NotImplementedError(f"lr policy {lr_policy} not implemented.")
    scheduler = {"scheduler": scheduler, "interval": "step"}
    return [optimizer], [scheduler]


class WarmupLinearSchedule(LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi *
                        float(self.cycles) * 2.0 * progress))
        )
