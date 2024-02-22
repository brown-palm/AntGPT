# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import torch
import os
import sys
import time
from typing import List

from transformers import LlamaTokenizer
from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model, load_llama_from_config
import pandas as pd
import numpy as np
import json
import time
import re
import pickle

PROMT_TEMPLATE = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and common sense most. Below is the observed 8 actions.\n\n### Observed actions: {}\n\n### Next 20 actions: "
PROMT_TEMPLATE_IDX = "Predict the next most possible 20 action indices in the format of verb noun index pair in chronological order that match the given observed 8 action indices and common sense most. Below is the observed 8 actions.\n\n### Observed actions: {}\n\n### Next 20 action indices: "

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    prompt_file: str=None,
    response_path: str=None,
    seed: int=42, #seed value for reproducibility
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        print('validation data path: ', prompt_file)
        print('Response saving path: ', response_path)
        if not os.path.exists(response_path):
            os.makedirs(response_path)
        if '.jsonl' in prompt_file:
            val_df = pd.read_json(prompt_file, lines=True)
            val_x = val_df['prompt'].apply(lambda x: x.replace("\n","").replace("#","")[:-1]).tolist()
            # val_y = val_df['completion'].apply(lambda x: x.strip().replace("\n","").replace("#","")[:-1]).tolist()
        elif '.csv' in prompt_file:
            val_df = pd.read_csv(prompt_file)
            val_x = val_df['prompt'].tolist()

        val_idx = np.arange(len(val_x)).tolist()
        clip_uids = val_df['clip_uid'].tolist()
        total_num = len(val_idx)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # tokenizer.add_special_tokens(
    #         {
    #             "pad_token": "<PAD>",
    #         }
    #     )
    # model.resize_token_embeddings(model.config.vocab_size + 1)
    
    # list all .pth files under the response_path
    response_files = os.listdir(response_path)
    response_files = [f for f in response_files if '.pth' in f]
    print("processed sample num: ", len(response_files))
    for ii, prompt_idx in enumerate(val_idx):
        clip_uid = clip_uids[prompt_idx]
        if clip_uid+'.pth' in response_files:
            continue
        example = val_x[prompt_idx].strip()
        if 'idx_' in prompt_file:
            user_prompt = PROMT_TEMPLATE_IDX.format(example)
        else:
            user_prompt = PROMT_TEMPLATE.format(example)
        # batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        batch = tokenizer(user_prompt, return_tensors="pt")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)                    
        logits = outputs.logits.cpu()
        print(str(ii+1)+'/'+str(total_num))
        torch.save(logits, os.path.join(response_path, clip_uid+'.pth'))

if __name__ == "__main__":
    fire.Fire(main)
