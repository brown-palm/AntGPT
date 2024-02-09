import numpy as np
import torch
import torch.distributed as dist
import itertools
import torchnet as tnt

def all_gather(data):
    data = data.contiguous()
    tensor_list = [torch.ones_like(data) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, data)
    return tensor_list


def distributed_topk_errors(preds, labels, ks, masks=None):
    """
    Computes the top-k error for each k. Average reduces the result with all other
    distributed processes.
    Args:
        preds (array): array of predictions. Dimension is NxC.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    preds = torch.cat(all_gather(preds), dim=0)
    labels = torch.cat(all_gather(labels), dim=0)
    if masks is not None:
        masks = torch.cat(all_gather(masks), dim=0)
        preds = preds[masks]
        labels = labels[masks]
    errors = topk_errors(preds, labels, ks)
    return errors

def distributed_topk_accs(preds, labels, ks, masks=None):
    """
    Computes the top-k accuracy for each k. Average reduces the result with all other
    distributed processes.
    Args:
        preds (array): array of predictions. Dimension is NxC.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    preds = torch.cat(all_gather(preds), dim=0)
    labels = torch.cat(all_gather(labels), dim=0)
    if masks is not None:
        masks = torch.cat(all_gather(masks), dim=0)
        preds = preds[masks]
        labels = labels[masks]
    accs = topk_accs(preds, labels, ks)
    return accs

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"

    # Find the top max_k predictions for each sample
    maxk = max(ks)
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, maxk, dim=1, largest=True, sorted=True
    )

    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum().item() for k in ks]
    return topks_correct

def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]

def topk_accs(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [x / preds.size(0) * 100.0 for x in num_topks_correct]

def mean_AP(preds, labels):
    '''
    preds: (B, #actions)
    labels: (B, #actions)
    '''

    apmeter = tnt.meter.APMeter()
    apmeter.add(preds,labels)
    mAP = apmeter.value().mean()
    return mAP


def distributed_mean_AP(preds, labels):
    '''
    preds: (B, #actions)
    labels: (B, #actions)
    '''
    preds = torch.cat(all_gather(preds), dim=0)
    labels = torch.cat(all_gather(labels), dim=0)
    return mean_AP(preds, labels)