import os
import torch
from .video_task import VideoTask
from ..utils import eval_util
from ..utils import file_util
from ..models.losses import MultiClassBCELoss

class VideoMultiClassClsTask(VideoTask):
    def __init__(self, cfg, steps_in_epoch):
        super().__init__(cfg, steps_in_epoch)
        self.label_mask = None

    def forward(self, batch):
        
        forecast_labels_idx = batch['forecast_labels_idx'] 

        input_texts = batch['text'] if self.cfg.data.use_gt_text else None   # (B, num_segments, 2)
        mask_text = batch['mask_text'] if self.cfg.data.use_gt_text else None

        image_features = batch['image_features'] if self.cfg.model.img_feat_size > 0 else None 
        mask_image = batch['mask_image'] if self.cfg.model.img_feat_size > 0 else None  # (B, num_images)

        input_pred_text = batch['text_feature'] if self.cfg.data.use_goal else None   # (B, Z, 2)
        mask_pred_text = batch['mask_text_feature'] if self.cfg.data.use_goal else None

        if self.label_mask is None:
            self.label_mask = batch['label_mask']

        logits = self.model.forward(input_texts, image_features, input_pred_text, mask_text, mask_image, mask_pred_text) 
        return logits

    def training_step(self, batch, batch_idx):
        step_results = {}

        labels = batch['forecast_labels_idx']   # (B, #actions)
        logits = self.forward(batch)[0][:, 0, :]         # logits: (B, #actions)
       
        loss, per_step_losses = self.loss_fn(logits, labels)
        step_results['loss'] = loss  # used to do backprop
        step_results['train/loss'] = loss
        self.log('train/loss_step', loss.item(), rank_zero_only=True)

        mAP = eval_util.distributed_mean_AP(logits.sigmoid()[:,self.label_mask[0]], labels[:,self.label_mask[0]])
        step_results[f'train/mAP'] = mAP

        return step_results

    def training_epoch_end(self, outputs):
        keys = [x for x in outputs[0].keys() if x != "loss"]
        for key in keys:
            metric = sum([x[key] for x in outputs]) / len(outputs)
            self.log(key, metric, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        labels = batch['forecast_labels_idx']   # (B, #actions)
        logits = self.forward(batch)[0][:, 0, :]          # logits: (B, #actions)

        step_results = {
            'labels': labels,
            'logits': logits,
        }
        
        loss, per_step_losses = self.loss_fn(logits, labels)
        self.log('val/loss', loss, batch_size=len(labels), sync_dist=True)

        return step_results

    def validation_epoch_end(self, outputs):
        keys = [x for x in outputs[0].keys()]

        data = {}
        for key in keys:
            data[key] = torch.cat([el[key] for el in outputs], dim=0)  # (N, #actions)
        
        mAP = eval_util.distributed_mean_AP(data['logits'].sigmoid()[:,self.label_mask[0]], data['labels'][:,self.label_mask[0]])
        f_mAP = eval_util.distributed_mean_AP(data['logits'].sigmoid()[:,self.label_mask[1]], data['labels'][:,self.label_mask[1]])
        r_mAP = eval_util.distributed_mean_AP(data['logits'].sigmoid()[:,self.label_mask[2]], data['labels'][:,self.label_mask[2]])
        
        self.log('val/mAP', mAP, sync_dist=True)
        self.log('val/mAP_frequent', f_mAP, sync_dist=True)
        self.log('val/mAP_rare', r_mAP, sync_dist=True)


    def test_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_epoch_end(self, outputs):
        raise NotImplementedError()
