# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import fire
import torch
import os
import sys
import pandas as pd
import numpy as np
import json
import time
from peft import PeftModel, PeftConfig
from transformers import (
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM
)
from vllm import LLM
from vllm import LLM, SamplingParams

def load_model(model_name, tp_size=1):

    llm = LLM(model_name, tensor_parallel_size=tp_size)
    return llm
PROMT_TEMPLATE = "Predict the next most possible 20 actions in the format of verb noun pair in chronological order that match the given observed 8 actions and common sense most. Below is the observed 8 actions.\n\n### Observed actions: {}\n\n### Prediction: "
def main(
    model_name: str,
    tp_size=1,
    n: int=5,
    max_new_tokens: int=200, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    response_path: str=None,
    seed: int=42, #seed value for reproducibility
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0.3, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
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
        elif '.csv' in prompt_file:
            val_df = pd.read_csv(prompt_file)
            val_x = val_df['prompt'].tolist()

        val_idx = np.arange(len(val_x)).tolist()
        total_num = len(val_idx)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_name, tp_size)
    sampling_param = SamplingParams(top_p=top_p, top_k=top_k, temperature=temperature, max_tokens=max_new_tokens, n=n)
    over = False
    while not over:
        try:
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
                example = val_x[prompt_idx].strip()
                user_prompt = PROMT_TEMPLATE.format(example)
                start_time = time.time()
                outputs = model.generate(user_prompt, sampling_params=sampling_param, use_tqdm=False)
                end_time = time.time()
                # print("time cost: ", end_time-start_time)
                res_list = []
                answer_len = []
                for jj in range(n):
                    output_text = outputs[0].outputs[jj].text
                    res_list.append(output_text)
                    try:
                        answer = output_text.strip('.').split(", ")
                        answer_len.append(len(answer))
                    except:
                        print('fail to parse')
                print(str(ii+1)+'/'+str(total_num), answer_len)
                print(res_list[0])
                responses_list.append(res_list)
                json.dump(responses_list, open(response_path, "w"))
            over = True
        except Exception as e:
            print(e)

if __name__ == "__main__":
    fire.Fire(main)
