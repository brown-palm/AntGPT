from transformers import LlamaTokenizer, LlamaForCausalLM
import json
import os
import pandas as pd
import numpy as np
import time
import argparse
import spacy
import torch
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model

message = [{'role':'system', 'content': 'You are a helpful assistant to figure out the scene based on the given actions.'}]

demos =[
       
        {
            "role": "user", "content": "take fork, wash fork, put fork, take knife, wash knife, take spoon, wash spoon, put spoon => clean tableware ###\n"
        },
        {
            "role": "user", "content": "remove pencil, put pencil, take pencil, put pencil, put pencil, take pencil, remove container, put container => writing ###\n"
        },
        {
            "role": "user", "content": "inspect gauge, take gauge, operate gauge, inspect gauge, put plier, take paper, clean gauge, put paper => fix car ###\n"
        },
        {
            "role": "user", "content": "clean flour, put cutter, put pot, take container, put container, close cabinet, move pot, remove tray => baking ###\n"
        },
        {
            "role": "user", "content": "clean floor, adjust table, put table, pull bucket, dip bucket, squeeze bucket, unfold napkin, clean napkin => set dining-table ###\n"
        },
        {
            "role": "user", "content": "move wheel, take chain, repair chain, remove chain, remove chain, put chain, put bicycle, put bicycle => fix bicycle ###\n"
        },
        {
            "role": "user", "content": "put plant, put plant, take plant, cut plant, put plant, take trowel, scoop bag, pour soil => gardening ###\n"
        },
        {
            "role": "user", "content": "lift container, hit container, put container, shake container, pour soil, take plant, put bucket, put soil => planting ###\n"
        },
        {
            "role": "user", "content": "take solder-iron, put solder-iron, hold wire, take solder-iron, attach wire, attach wire, put solder-iron, take rubber-band => solder wires ###\n"
        },
        {
            "role": "user", "content": "put iron, adjust shirt, take iron, press shirt, put iron, adjust shirt, take shirt, put cloth => iron clothes ###\n"
        },
        {
            "role": "user", "content": "take flower, take flower, take flower, take flower, take flower, take flower, take flower, cut flower => arrange flowers ###\n"
        },
        {
            "role": "user", "content": "take broom, move broom, take dustpan, put dustpan, move broom, take dustpan, clean trash, put broom => clean room ###\n"
        },
]

def llama_v2_prompt(messages: list[dict]):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    # B_INST, E_INST = "", ""
    # B_SYS, E_SYS = "", ""
    # BOS, EOS = "", ""
    
    DEFAULT_SYSTEM_PROMPT = f"""'You are a helpful assistant to figure out the scene based on the given actions."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Llama-2-13b-chat-hf", help="name of the model")
    parser.add_argument('--response_dir', type=str, default="output", help="path to the output directory")
    parser.add_argument('--dataset_dir', type=str, default="dataset", help="path to the dataset directory")
    parser.add_argument('--val_name', type=str, default="val_nseg8_recog_subset600.jsonl", help="name of the validation file")
    parser.add_argument('--response_name', type=str, default="goal_val600_12sample_full.json", help="name of the output file")
    parser.add_argument('--temperature', type=float, default=0.5, help="GPT api parameter temperature")
    # parser.add_argument('--n', type=int, default=1, help="GPT api parameter n")
    parser.add_argument('--max_tokens', type=int, default=500, help="GPT api parameter max_tokens")
    parser.add_argument('--max_seq_len', type=int, default=4096, help="max sequence length")
    parser.add_argument('--max_batch_size', type=int, default=5, help="max batch size")
    parser.add_argument('--max_gen_len', type=int, default=100, help="max generation length")
    parser.add_argument('--num_samples', type=int, default=12, help="number of sample paths")
    parser.add_argument('--top_p', type=float, default=0.9, help="GPT api parameter top_p")
    args = parser.parse_args()

   

    # tokenizer_path = '/gpfs/data/superlab/models/llama2/llama/checkpoints/original/tokenizer.model'
    ckpt_dir = os.path.join('/gpfs/data/superlab/models/llama2/llama/checkpoints/hf', args.model_name)
    model = load_model(ckpt_dir, quantization=False)
    tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir, legacy=False)
    # tokenizer.add_special_tokens(
    #         {
    #             "pad_token": "<PAD>",
    #         }
    #     )
    model.eval()

    # generator = LLM(model="meta-llama/Llama-2-13b-hf")
    # generator = LLM(model=ckpt_dir)
    # generator = LLM(model="/users/qzhao25/llama/7b-ek")

    

    # Print the outputs.
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


    response_path = os.path.join(args.response_dir, args.response_name)
    val_path = os.path.join(args.dataset_dir, args.val_name)
    print('validation data path: ', val_path)
    print('Response saving path: ', response_path)

    val_df = pd.read_json(val_path, lines=True)
    val_x = val_df['prompt'].apply(lambda x: x.replace("\n","").replace("#","")[:-1]).tolist()
    val_y = val_df['completion'].apply(lambda x: x.strip().replace("\n","").replace("#","")[:-1]).tolist()


    val_idx = np.arange(len(val_x)).tolist()
    total_num = len(val_idx)
    over = False
    total_non_unique = 0
    print("start ICL querying from Llama{}".format(args.model_name))
    while not over:
        try:
            try:
                responses_list = json.load(open(response_path, "r"))
            except:
                responses_list = []
                json.dump(responses_list, open(response_path, "w"))
            print("processed sample num: ", len(responses_list)) 

            for ii, prompt_idx in enumerate(val_idx):
                if ii < len(responses_list):
                    continue
                
                ## initialize list to start voting 
                goal_list = []

                ## ITERATE THROUGH NUMBER OF SAMPLES YOU WANT
                # print("Samples")
                for _ in range(args.num_samples):
                    prompt_message = 'Suppose a person has performed the given actions in the form of a sequence of action pairs. Each action pair is defined by a {verb} and a {noun}, separated by a space. What is the most possible scene according to the given previous actions? Answer the scene using no more than three words. Please only output ONE total scene. Here are some demonstrations and the questions: ###\n'
                    for demo in demos:
                        prompt_message += demo['content']
                    prompt_message += val_x[prompt_idx] + " => "
                    mes = [{'role':'user', 'content': prompt_message}]

                    dialogs = llama_v2_prompt(mes)
                    # dialogs = [mes] 
                    # print("Dialogs: ", dialogs)
                    batch = tokenizer(dialogs, return_tensors="pt")
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    with torch.no_grad():
                        start_time = time.time()
                        outputs = model.generate(
                            **batch,
                            max_new_tokens=50,
                            do_sample=True,
                            top_p=1.0,
                            temperature=0.3,
                            min_length=None,
                            use_cache=True,
                            top_k=50,
                            repetition_penalty=1.0,
                            length_penalty=1,
                            num_return_sequences = 5,
                        )
                    end_time = time.time()
                    # print("time cost: ", end_time-start_time)
                    res_list = []
                    answer_len = []
                    for jj in range(len(outputs)):
                        output_text = tokenizer.decode(outputs[jj], skip_special_tokens=True).split("=> [/INST]")[-1] 
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
            print("Error")
            ## reset counter
            # total_non_unique = 0
            # time.sleep(30)
            