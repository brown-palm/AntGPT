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

PROMT_START = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and common sense most. Below is the observed 8 actions.\n\n### Observed actions:\n"
PROMT_END = "{}\n\n### Prediction:"
class TinyVisLlama(torch.nn.Module):
    def __init__(self, llama_config, vis_size):
        super().__init__()
        self.llama = LlamaForCausalLM(llama_config)
        self.proj = torch.nn.Linear(vis_size, llama_config.hidden_size, bias=False)

class VisLlama(torch.nn.Module):
    def __init__(self, llama, llama_config, vis_size):
        super().__init__()
        self.llama = llama
        self.proj = torch.nn.Linear(vis_size, llama_config.hidden_size, bias=False)

        
def main(
    model_name,
    use_vis: bool,
    pure_vis: bool=False,
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
    print_vis_emb: bool = True,
    **kwargs
):
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
            clip_uids = val_df['clip_uid'].tolist()
        elif '.csv' in prompt_file:
            val_df = pd.read_csv(prompt_file)
            val_x = val_df['prompt'].tolist()
            if 'caption' in prompt_file:
                captions = val_df['caption'].tolist()
            # val_y = val_df['completion'].tolist()
            clip_uids = val_df['clip_uid'].tolist()

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
        model = TinyVisLlama(llama_config=configuration, vis_size=768)
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
        llama = load_model(model_name, quantization)
        if peft_model:
            llama = load_peft_model(llama, peft_model)
        model = VisLlama(llama, llama.config, vis_size=768)
        proj_path = os.path.join(peft_model, 'proj.pt')
        embed_tokens_path = os.path.join(peft_model, 'embed_tokens.pt')
        model.proj.load_state_dict(torch.load(proj_path))
        model.llama.model.model.embed_tokens.load_state_dict(torch.load(embed_tokens_path))
        model.to("cuda")

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
    # model.resize_token_embeddings(model.config.vocab_size + 1)
    
    answers_list = []
    answer_len_list = []
    goals_list = []
    try:
        responses_list = json.load(open(response_path, "r"))
    except:
        responses_list = []
        json.dump(responses_list, open(response_path, "w"))
    print("processed sample num: ", len(responses_list))
    for ii, prompt_idx in enumerate(val_idx):
        if ii < len(responses_list):
            continue
        # batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        # batch = tokenizer(user_prompt, return_tensors="pt")
        clip_uid = clip_uids[prompt_idx]
        feat_name = clip_uid.split("_")[0] + '.pt'
        if 'test' in prompt_file:
            obs_feats = torch.load('/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_ego4dv2/'+feat_name)
            indices_file = '/users/swang299/code/AntGPT-arxiv/frame_ids/clip/test/' + clip_uid + '.txt'
        elif 'v2' in prompt_file:
            obs_feats = torch.load('/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_ego4dv2/'+feat_name)
            indices_file = '/users/swang299/code/AntGPT-arxiv/frame_ids/clip/v2/' + clip_uid + '.txt'
        else:
            obs_feats = torch.load('/gpfs/data/csun45/cfu17/GLIP_temp/output_CLIP_img_embedding_kmeans_ego4d/'+feat_name)
            indices_file = '/users/swang299/code/AntGPT-arxiv/frame_ids/clip/v1_new/' + clip_uid + '.txt'
        
        with open(indices_file) as f:
            indices = f.readline().strip().split(' ')
        indices = [int(i) for i in indices]
        obs_feats = obs_feats[indices].unsqueeze(0).to("cuda")
        
        start = time.perf_counter()
        if '.pt' in model_name:
            embed_tokens = model.llama.model.embed_tokens
        else:
            embed_tokens = model.llama.model.model.embed_tokens
        with torch.no_grad():
            if pure_vis:
                input_ = ''
            elif 'caption_only' in prompt_file:
                input_ = '\n' + captions[prompt_idx].strip()
            elif 'caption' in prompt_file:
                input_ = '\n' + captions[prompt_idx].strip() + '\n' + val_x[prompt_idx].strip()
            else:
                input_ = '\n' + val_x[prompt_idx].strip()
            if use_vis:
                obs_embeds = model.proj(obs_feats)
                start_ids = tokenizer(PROMT_START, return_tensors="pt")["input_ids"].to("cuda")
                start_embeds = embed_tokens(start_ids)
                
                # label = val_y[prompt_idx].strip()
                # print("label: ", label)
                # label = tokenizer("take",return_tensors="pt")["input_ids"][:, 1:].to("cuda")
                # label_embeds = model.embed_tokens(label)     

                prompt_ids = tokenizer(PROMT_END.format(input_), return_tensors="pt")["input_ids"][:, 1:].to("cuda")
                prompt_embeds = embed_tokens(prompt_ids)
                
                inputs_embeds = torch.cat([start_embeds, obs_embeds, prompt_embeds], dim=1)
                batch = {"inputs_embeds": inputs_embeds, "attention_mask": torch.ones(1, inputs_embeds.shape[1], dtype=torch.int64).to("cuda")}
                # batch = {k: v.to(model.device) for k, v in batch.items()}
            else:
                
                prompt = PROMT_START + PROMT_END.format(input_)
                inputs_embeds = embed_tokens(tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda"))
                batch = {"inputs_embeds": inputs_embeds, "attention_mask": torch.ones(1, inputs_embeds.shape[1], dtype=torch.int64).to("cuda")}

            outputs = model.llama.generate(
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
            
            # vis_tokens = model.llama.lm_head(inputs_embeds)
            # vis_tokens = torch.argmax(vis_tokens, dim=-1)
            # print("inputs: ", tokenizer.batch_decode(vis_tokens.detach().cpu().numpy(), skip_special_tokens=True)[0])
            if print_vis_emb and use_vis:
                # logits = model.llama(**batch).logits
                # start_len = start_embeds.shape[1]
                # obs_len = obs_embeds.shape[1]
                # logits = logits[:, start_len: start_len+obs_len, :]
                # print(logits.shape)
                # preds = torch.argmax(logits, -1)
                # print(tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=False)[0])
                
                # use embed_tokens to decode the vision embeddings
                obs_tokens = obs_embeds @ embed_tokens.weight.T
                print(obs_tokens.shape)
                obs_tokens = torch.argmax(obs_tokens, dim=-1)
                print(tokenizer.batch_decode(obs_tokens.detach().cpu().numpy(), skip_special_tokens=False))
        
        e2e_inference_time = (time.perf_counter()-start)
        print(f"the inference time is {e2e_inference_time} s")
        res_list = []
        answer_len = []
        for jj in range(len(outputs)):
            output_text = tokenizer.decode(outputs[jj], skip_special_tokens=True) 
            res_list.append(output_text)
            try:
                answer = output_text.strip('.').split(", ")
                answer_len.append(len(answer))
            except:
                print('fail to parse')
        print(str(ii+1)+'/'+str(total_num), answer_len)
        # delete characters not in [a-z], [A-Z] and [, ]
        # tmp = re.sub(r'[^a-zA-Z, ]+', '', res_list[0])
        # delete characters nont in [a-z], [A-Z] in the start of the string
        # tmp = re.sub(r'^[^a-zA-Z]+', '', tmp)
        # print(tmp)
        print(res_list[0])
        responses_list.append(res_list)
        json.dump(responses_list, open(response_path, "w"))


if __name__ == "__main__":
    fire.Fire(main)
