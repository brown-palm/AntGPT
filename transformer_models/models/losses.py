import torch
from torch import nn


def get_loss(cfg):
    loss_name = cfg.model.loss_fn
    if loss_name == 'LTAWeightedLoss':
        return LTAWeightedLoss(cfg.model.loss_wts_heads, cfg.model.loss_wts_temporal)
    elif loss_name == 'MultiClassBCELoss':
        return MultiClassBCELoss()
    

class LTAWeightedLoss(nn.Module):
    def __init__(self, loss_wts_heads, loss_wts_temporal):
        super(LTAWeightedLoss, self).__init__()
        # self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.loss_fun = nn.CrossEntropyLoss(reduction='mean')
        self.loss_wts_heads = loss_wts_heads
        self.loss_wts_temporal = loss_wts_temporal

    def forward(self, logits, targets, mask=None):
        '''
        logits: [(B, Z, #verbs), (B, Z, #nouns)]
        targets: (B, Z, 2)
        mask: (B, Z) or None
        '''
        # loss_wts_heads = self.loss_wts_heads
        # loss_wts_temporal = torch.tensor(self.loss_wts_temporal, device=logits[0].device, dtype=logits[0].dtype)  # (Z, )
        # loss = 0.
        # per_step_losses = []  # [(Z,), (Z,)]
        # for head_idx in range(len(logits)):
        #     batch_size, num_actions_to_predict, num_classes = logits[head_idx].shape
        #     logits_head = logits[head_idx].reshape((-1, num_classes))  # (B*Z, #xxx)
        #     targets_head = targets[..., head_idx].reshape((-1,))  # (B*Z)
        #     loss_head = self.cross_entropy(logits_head, targets_head)  # (B*Z)
        #     loss_head = loss_head.reshape(batch_size, num_actions_to_predict)  # (B, Z)
        #     per_step_losses.append(torch.mean(loss_head, dim=0).detach())  # (Z)
        #     loss += loss_wts_heads[head_idx] * torch.sum(loss_head * loss_wts_temporal) / batch_size
        # return loss, per_step_losses
        loss_wts = self.loss_wts_heads
        losses = [0, 0]
        for head_idx in range(len(logits)):
            #if head_idx == 0: continue
            pred_head = logits[head_idx]  # (B, Z, C)
            for seq_idx in range(pred_head.shape[1]):
                losses[head_idx] += loss_wts[head_idx] * self.loss_fun(
                    pred_head[:, seq_idx], targets[:, seq_idx, head_idx]
                )
        return sum(losses), None
    

class MultiClassBCELoss(nn.Module):
    def __init__(self):
        super(MultiClassBCELoss, self).__init__()
        self.loss_func = nn.BCELoss()
    def forward(self, logits, targets, mask=None):
        '''
        logits: (B, #actions)
        targets: (B, #actions)
        '''
        loss = []
        targets = targets.to(torch.float32)
        for i in range(logits.shape[-1]):
            loss.append(self.loss_func(logits[:,i].sigmoid(), targets[:,i]))
        return sum(loss) / len(loss), None
