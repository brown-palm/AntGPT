# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from functools import partial

from ft_datasets import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
    get_ego4d_lta_dataset,
    get_ek_dataset,
    get_vis_ego4d_lta_dataset,
)
from typing import Optional


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=224),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "ego4d_v1": get_ego4d_lta_dataset,
    "ego4d_v2": get_ego4d_lta_dataset,
    "ego4d_v1_aug": get_ego4d_lta_dataset,
    "ego4d_v2_aug": get_ego4d_lta_dataset,
    "ego4d_v1_recog": get_ego4d_lta_dataset,
    "ego4d_v2_recog": get_ego4d_lta_dataset,
    "ego4d_v2_3in1": get_ego4d_lta_dataset,
    "ego4d_v2_aug_egovlp": get_ego4d_lta_dataset,
    "idx_ego4d_v1": get_ego4d_lta_dataset,
    "idx_ego4d_v2": get_ego4d_lta_dataset,
    "idx_ego4d_v1_aug": get_ego4d_lta_dataset,
    "idx_ego4d_v2_aug": get_ego4d_lta_dataset,
    "idx_ego4d_v1_recog": get_ego4d_lta_dataset,
    "idx_ego4d_v2_recog": get_ego4d_lta_dataset,
    "idx_ego4d_v2_3in1": get_ego4d_lta_dataset,
    "idx_ego4d_v2_aug_egovlp": get_ego4d_lta_dataset,
    "ek_dataset": get_ek_dataset,
    "ek_dataset_si": get_ek_dataset,
    "gaze_dataset": get_ek_dataset,
    "gaze_dataset_si": get_ek_dataset,
    "vis_ego4d_v1": get_vis_ego4d_lta_dataset,
    "vis_ego4d_v1_recog": get_vis_ego4d_lta_dataset,
    "vis_ego4d_v1_aug": get_vis_ego4d_lta_dataset,
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train", use_goal: Optional[bool] = False
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    if 'ego4d' in dataset_config.dataset:
        return DATASET_PREPROC[dataset_config.dataset](
            dataset_config,
            tokenizer,
            get_split(),
            use_goal,
        )
    else:
        return DATASET_PREPROC[dataset_config.dataset](
            dataset_config,
            tokenizer,
            get_split(),
        )
