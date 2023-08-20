import numpy as np
import torch
import editdistance
from sklearn.metrics import average_precision_score
import torch.distributed as dist
import itertools
import torchnet as tnt
from . import file_util


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

def edit_distance(preds, labels):
    """
    Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein
    Lowest among K predictions
    Average over N
    """
    N, Z, K = preds.shape
    dists = []
    for n in range(N):
        dist = min([editdistance.eval(preds[n, :, k], labels[n])/Z for k in range(K)])
        dists.append(dist)
    return np.mean(dists)

def distributed_edit_distance(preds, labels):
    preds = torch.cat(all_gather(preds), dim=0)
    labels = torch.cat(all_gather(labels), dim=0)
    return 50(preds, labels)

def AUED(preds, labels):
    '''
    average over N
    '''
    N, Z, K = preds.shape
    preds = preds.cpu().numpy()  # (N, Z, K)
    labels = labels.cpu().numpy()  # (N, Z)
    # ED = np.vstack(
    #     [edit_distance(preds[:, :z], labels[:, :z]) for z in range(1, Z + 1)]
    # )
    ED = np.array([edit_distance(preds[:, :z], labels[:, :z]) for z in range(1, Z + 1)])
    AUED = np.trapz(y=ED, axis=0) / (Z - 1) if Z > 1 else 1.0

    output = {"AUED": AUED}
    output.update({f"ED_{z}": ED[z] for z in range(Z)})
    return output

def distributed_AUED(preds, labels):
    '''
    preds: (B, Z, K)
    labels: (B, Z)
    '''
    preds = torch.cat(all_gather(preds), dim=0)
    labels = torch.cat(all_gather(labels), dim=0)
    return AUED(preds, labels)

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


def check_test_submit(path):
    submit = file_util.load_json(path)
    idx_set = set(submit.keys())

    gt_path = "data/ego4d/annotations/fho_lta_test_unannotated.json"
    gt_data = file_util.load_json(gt_path)
    clip_info_ls = gt_data["clips"]
    for clip_uid, clip_info in itertools.groupby(clip_info_ls, lambda x: x["clip_uid"]):
        clip_info = list(clip_info)
        clip_info.sort(key=lambda x: x["action_idx"])
        for i in range(7, len(clip_info) - 20):
            clip_action_idx = f"{clip_uid}_{i}"
            if clip_action_idx not in idx_set:
                print("missing key: ", clip_action_idx)
    print("finish")