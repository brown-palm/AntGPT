import datasets
from .utils import Concatenator
import pandas as pd
import numpy as np
import copy
import torch
import os
from torch.utils.data import Dataset
from datasets import load_dataset
IGNORE_INDEX = -100
PROMT_TEMPLATE = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and common sense most. Below is the observed 8 actions.\n\n### Observed actions: {}\n\n### Next 20 actions: "
PROMT_TEMPLATE_IDX = "Predict the next most possible 20 action indices in the format of verb noun index pair in chronological order that match the given observed 8 action indices and common sense most. Below is the observed 8 actions.\n\n### Observed actions: {}\n\n### Next 20 action indices: "
PROMT_TEMPLATE_GOAL = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and scene and common sense most. Below is the scene and observed 8 actions.\n\n### Scene: {}\n\n### Observed actions: {}\n\n### Next 20 actions: "
PROMT_TEMPLATE_OLD = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and common sense most. Below is the observed 8 actions.\n\n### Observed actions: {}\n\n### Prediction: "

class Ego4D_LTA(Dataset):
    def __init__(
        self,
        tokenizer,
        csv_path,
        use_goal,
        max_words = 250
        ):
        self.max_words = max_words
        self.split = 'train'
        self.dataset = load_dataset("csv", data_files={self.split: [csv_path],}, delimiter=",")
        self.tokenizer = tokenizer
        self.use_goal = use_goal
        if 'idx_' in csv_path:
            self.PROMT_TEMPLATE = PROMT_TEMPLATE_IDX
        elif use_goal:
            self.PROMT_TEMPLATE = PROMT_TEMPLATE_GOAL
        else:
            self.PROMT_TEMPLATE = PROMT_TEMPLATE_OLD

    def __len__(self):
        return self.dataset[self.split].shape[0]

    def convert_to_features(self, example_batch):

        # Create prompt and tokenize contexts and questions

        input_ = example_batch["prompt"].strip()
        target_ = example_batch["completion"].strip()
        if self.use_goal:
            goal_ = example_batch["goal"].strip()
            # prompt = self.PROMT_TEMPLATE.format(goal_, input_)
            prompt = self.PROMT_TEMPLATE.format(goal_, '')
            # print(prompt)
        else:
            prompt = self.PROMT_TEMPLATE.format(input_)
        example = prompt + target_
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = IGNORE_INDEX
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64)))
            labels = torch.cat((labels, IGNORE_INDEX*torch.ones(padding, dtype=torch.int64)))
        elif padding < 0:
            example = example[: self.max_words]
            labels = labels[: self.max_words]
        example_mask = example.ne(self.tokenizer.pad_token_id)
        # example_mask = example.ne(0)
        # label_mask = labels.ge(0)
        # example[~example_mask] = 0
        # labels[~label_mask] = 0
        example_mask = example_mask.float()
        # label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }

    def __getitem__(self, index):
        sample = self.convert_to_features(self.dataset[self.split][index])
        return sample


def get_dataset(
    dataset_config, tokenizer, csv_path, use_goal):
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    dataset = Ego4D_LTA(
        tokenizer=tokenizer,
        csv_path=csv_path,
        use_goal=use_goal,
    )
    return dataset