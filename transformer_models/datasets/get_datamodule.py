from .base_dataset import BaseDataModule
from .gaze_lta_dataset import GazeLTADataModule


def get_dm(cfg):
    if cfg.multicls.enable:
        return GazeLTADataModule(cfg)
    else:
        return BaseDataModule(cfg)