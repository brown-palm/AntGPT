import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from . import BaseVideoDataset
from ..utils.parser import parse_args, load_config


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if not hasattr(self, 'train_set'):
                self.train_set = BaseVideoDataset(self.cfg, self.cfg.data.train_anno_path, True, False)
            if not hasattr(self, 'val_set'):
                self.val_set = BaseVideoDataset(self.cfg, self.cfg.data.val_anno_path, False, False)

        if stage == "test" or stage is None:
            self.test_set = BaseVideoDataset(self.cfg,self.cfg.data.test_anno_path, False, True)

    def train_dataloader(self):
        if not hasattr(self, 'train_loader'):
            num_gpus = self.cfg.num_gpus
            assert self.cfg.train.batch_size % num_gpus == 0
            batch_size = self.cfg.train.batch_size // num_gpus
            self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size=batch_size, num_workers=self.cfg.train.num_workers)
        return self.train_loader

    def val_dataloader(self):
        if not hasattr(self, 'val_loader'):
            num_gpus = self.cfg.num_gpus
            assert self.cfg.val.batch_size % num_gpus == 0
            batch_size = self.cfg.val.batch_size // num_gpus
            self.val_loader = DataLoader(self.val_set, shuffle=False, batch_size=batch_size, num_workers=self.cfg.val.num_workers)
        return self.val_loader

    def test_dataloader(self):
        if not hasattr(self, 'test_loader'):
            # num_gpus = self.cfg.num_gpus
            num_gpus = 1
            assert self.cfg.test.batch_size % num_gpus == 0
            batch_size = self.cfg.test.batch_size // num_gpus
            self.test_loader = DataLoader(self.test_set, shuffle=False, batch_size=batch_size, num_workers=self.cfg.test.num_workers, drop_last=False)
        return self.test_loader

    
    
def sanity_check():
    args = parse_args()
    cfg = load_config(args)
    dm = BaseDataModule(cfg)
    dm.setup(stage="fit")

    for annotation in dm.val_set:
        if torch.lt(annotation['text'], 0.0).sum() > 0:
            print(annotation)


if __name__ == '__main__':
    sanity_check()
    # main()


'''
python -m src.datasets.base_dataset --cfg configs/ego4d/recognition_sf_video.yaml --exp_name ego4d/random \
    val.batch_size 1

python -m src.datasets.base_dataset --cfg configs/ego4d/text.yaml --exp_name ego4d/random \
    val.batch_size 1 train.batch_size 1
'''