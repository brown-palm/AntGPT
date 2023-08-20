import torch
from ..optimizers import lr_scheduler
from ..models.model import ClassificationModule
from pytorch_lightning.core import LightningModule
from ..models.losses import get_loss


class VideoTask(LightningModule):
    def __init__(self, cfg, steps_in_epoch):
        super().__init__()
        self.cfg = cfg
        self.steps_in_epoch = steps_in_epoch
        self.save_hyperparameters()
        self.model = self.build_model()
        self.loss_fn = get_loss(cfg)

    def build_model(self):
        if self.cfg.model.model == 'classification':
            model = ClassificationModule(self.cfg)
            return model
        else:
            raise NotImplementedError(f'model {self.cfg.model.model} not implemmented')

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        # steps_in_epoch = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        return lr_scheduler.lr_factory(self.model, self.cfg, self.steps_in_epoch)
