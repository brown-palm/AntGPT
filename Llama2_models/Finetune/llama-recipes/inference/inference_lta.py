# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import torch
import os
import sys
import time
from typing import List

from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model, load_llama_from_config
import pandas as pd
import numpy as np
import json
import time
import re

PROMT_TEMPLATE = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and common sense most. Below is the observed 8 actions.\n\n### Observed actions: {}\n\n### Next 20 actions: "
PROMT_TEMPLATE_IDX = "Predict the next most possible 20 action indices in the format of verb noun index pair in chronological order that match the given observed 8 action indices and common sense most. Below is the observed 8 actions.\n\n### Observed actions: {}\n\n### Next 20 action indices: "
PROMT_TEMPLATE_GOAL = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and scene and common sense most. Below is the scene and observed 8 actions.\n\n### Scene: {}\n\n### Observed actions: {}\n\n### Next 20 actions: "
PROMT_TEMPLATE_OLD = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and common sense most. Below is the observed 8 actions.\n\n### Observed actions: {}\n\n### Prediction: "

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens: int=200, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    response_path: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0.3, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    if response_path:
        use_goal = 'goal' in response_path
    else:
        use_goal = False
    print('use_goal: ', use_goal)
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        print('validation data path: ', prompt_file)
        print('Response saving path: ', response_path)

        if '.jsonl' in prompt_file:
            val_df = pd.read_json(prompt_file, lines=True)
            val_x = val_df['prompt'].apply(lambda x: x.replace("\n","").replace("#","")[:-1]).tolist()
            # val_y = val_df['completion'].apply(lambda x: x.strip().replace("\n","").replace("#","")[:-1]).tolist()
            if use_goal:
                goals_list = val_df['goal'].tolist()
        elif '.csv' in prompt_file:
            val_df = pd.read_csv(prompt_file)
            val_x = val_df['prompt'].tolist()
            if use_goal:
                goals_list = val_df['goal'].tolist()
        val_idx = np.arange(len(val_x)).tolist()
        total_num = len(val_idx)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    if '.pt' in model_name:
        configuration = LlamaConfig(vocab_size = 32000,
                                    hidden_size = 768,
                                    intermediate_size = 2048,
                                    num_hidden_layers = 6,
                                    num_attention_heads = 6,
                                    num_key_value_heads = 6,
                                    max_position_embeddings = 300,
                                    )
        model = LlamaForCausalLM(configuration)
        model.to('cuda')
        model.load_state_dict(torch.load(model_name))
        
        file_name = model_name.split('/')[-1]
        file_prefix_path = model_name.replace(file_name, '')
        
        if "test" in prompt_file:
            response_path = file_prefix_path + "test_"+file_name.replace('.pt', '.json')
        elif "train" in prompt_file:
            response_path = file_prefix_path + "train_"+file_name.replace('.pt', '.json')
        elif "val" in prompt_file:
            response_path = file_prefix_path + "val_"+file_name.replace('.pt', '.json')
    else:
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

    tokenizer = LlamaTokenizer.from_pretrained('/gpfs/data/superlab/models/llama2/llama/checkpoints/hf/Llama-2-7b-hf')
    # tokenizer.add_special_tokens(
    #         {
    #             "pad_token": "<PAD>",
    #         }
    #     )
    model.resize_token_embeddings(model.config.vocab_size + 1)
    
    answers_list = []
    answer_len_list = []
    try:
        responses_list = json.load(open(response_path, "r"))
    except:
        responses_list = []
        json.dump(responses_list, open(response_path, "w"))
    print("processed sample num: ", len(responses_list))
    for ii, prompt_idx in enumerate(val_idx):
        if ii < len(responses_list):
            continue
        example = val_x[prompt_idx].strip()
        if 'idx_' in prompt_file:
            user_prompt = PROMT_TEMPLATE_IDX.format(example)
        elif use_goal:
            goal = goals_list[prompt_idx].strip()
            user_prompt = PROMT_TEMPLATE_GOAL.format(goal, example)
        else:
            user_prompt = PROMT_TEMPLATE_OLD.format(example)
        # batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        batch = tokenizer(user_prompt, return_tensors="pt")
        # print(batch)
        # batch = model.model.embed_tokens(batch['input_ids'].to(model.device))
        # batch = {"inputs_embeds": batch}
        batch = {k: v.to(model.device) for k, v in batch.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences = 5,
                pad_token_id = tokenizer.eos_token_id,
                **kwargs 
            )
        e2e_inference_time = (time.perf_counter()-start)
        print(f"the inference time is {e2e_inference_time} s")
        res_list = []
        answer_len = []
        for jj in range(len(outputs)):
            if 'idx_' in prompt_file:
                output_text = tokenizer.decode(outputs[jj], skip_special_tokens=True).split("20 action indices: ")[-1] 
            else:
                output_text = tokenizer.decode(outputs[jj], skip_special_tokens=True).split("Prediction: ")[-1]
            res_list.append(output_text)
            try:
                answer = output_text.strip('.').split(", ")
                answer_len.append(len(answer))
            except:
                print('fail to parse')
        print(str(ii+1)+'/'+str(total_num), answer_len)
        # delete characters not in [a-z], [A-Z] and [, ]
        tmp = re.sub(r'[^a-zA-Z, ]+', '', res_list[0])
        # delete characters nont in [a-z], [A-Z] in the start of the string
        tmp = re.sub(r'^[^a-zA-Z]+', '', tmp)
        print(tmp)
        # print(res_list[0])
        responses_list.append(res_list)
        json.dump(responses_list, open(response_path, "w"))


if __name__ == "__main__":
    fire.Fire(main)
