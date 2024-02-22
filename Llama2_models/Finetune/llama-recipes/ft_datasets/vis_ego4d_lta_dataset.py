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
# PROMT_TEMPLATE = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and common sense most. Below is the observed 8 actions.\n\n### Observed actions: {}\n\n### Next 20 actions: "
# PROMT_TEMPLATE_IDX = "Predict the next most possible 20 action indices in the format of verb noun index pair in chronological order that match the given observed 8 action indices and common sense most. Below is the observed 8 actions.\n\n### Observed actions: {}\n\n### Next 20 action indices: "
# PROMT_TEMPLATE_GOAL = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and scene and common sense most. Below is the scene and observed 8 actions.\n\n### Scene: {}\n\n### Observed actions: {}\n\n### Next 20 actions: "
PROMT_START = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and common sense most. Below is the observed 8 actions.\n\n### Observed actions:\n"
PROMT_END = "{}\n\n### Prediction:"

class Vis_Ego4D_LTA(Dataset):
    def __init__(
        self,
        tokenizer,
        csv_path,
        use_vis,
        pure_vis,
        max_words = 300
        ):
        self.max_words = max_words
        self.split = 'train'
        self.dataset = load_dataset("csv", data_files={self.split: [csv_path],}, delimiter=",")
        self.tokenizer = tokenizer
        self.use_vis = use_vis
        self.pure_vis = pure_vis
        self.csv_path = csv_path
        
        if use_vis:
            if 'caption_only' in csv_path:
                self.max_words = 300
            elif 'caption' in csv_path:
                self.max_words = 320


    def __len__(self):
        return self.dataset[self.split].shape[0]

    def convert_to_features(self, example_batch):

        # Create prompt and tokenize contexts and questions
        clip_uid_ = example_batch["clip_uid"].strip()
        if self.pure_vis:
            input_ = ''
        elif 'caption_only' in self.csv_path:
            input_ = '\n' + example_batch["caption"].strip()
        elif 'caption' in self.csv_path:
            input_ = '\n' + example_batch["caption"].strip() + '\n' + example_batch["prompt"].strip() 
        else:
            input_ = '\n' + example_batch["prompt"].strip()
        target_ = example_batch["completion"].strip()
        feat_name = clip_uid_.split("_")[0] + '.pt'
        if 'v2' in self.csv_path:
            obs_feats_ = torch.load('/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_ego4dv2/'+feat_name)
            indices_file = '/users/swang299/code/AntGPT-arxiv/frame_ids/clip/v2/' + clip_uid_ + '.txt'
        else:
            obs_feats_ = torch.load('/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_kmeans_ego4d/'+feat_name)
            indices_file = '/users/swang299/code/AntGPT-arxiv/frame_ids/clip/v1_new/' + clip_uid_ + '.txt'
        # read the first line
        with open(indices_file) as f:
            indices = f.readline().strip().split(' ')
        indices = [int(i) for i in indices]
        obs_feats_ = obs_feats_[indices]  
        
        if self.use_vis:
            dummy_prompt = PROMT_START + obs_feats_.shape[0] * self.tokenizer.eos_token + PROMT_END.format(input_)
        else:
            dummy_prompt = PROMT_START + PROMT_END.format(input_)
        example = dummy_prompt + target_
        # print(obs_feats_.shape)
        dummy_prompt = torch.tensor(
            self.tokenizer.encode(dummy_prompt),
            dtype=torch.int64,
        )
        # prompt_ids = copy.deepcopy(dummy_prompt)[obs_feats_.shape[0]+1:]
        
        example = self.tokenizer.encode(example)
        # change all the eos_token to -1
        example = [x if x != self.tokenizer.eos_token_id else -1 for x in example]
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(dummy_prompt)] = IGNORE_INDEX
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64)))
            labels = torch.cat((labels, IGNORE_INDEX*torch.ones(padding, dtype=torch.int64)))
        elif padding < 0:
            example = example[: self.max_words]
            labels = labels[: self.max_words]
        
        example_mask = example.ne(self.tokenizer.pad_token_id)
        # example_mask = example.ne(0)
        # example_mask = example.ne(-1)
        example_mask = example_mask.float()
        
        return {
            "input_ids": example,
            "obs_feats": obs_feats_,
            "labels": labels,
            "attention_mask":example_mask,
        }

    def __getitem__(self, index):
        sample = self.convert_to_features(self.dataset[self.split][index])
        return sample


def get_dataset(
    dataset_config, tokenizer, csv_path, use_vis, pure_vis):
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    dataset = Vis_Ego4D_LTA(
        tokenizer=tokenizer,
        csv_path=csv_path,
        use_vis=use_vis,
        pure_vis=pure_vis,
    )
    return dataset