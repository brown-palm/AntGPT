import json
import os
import pandas as pd
import numpy as np
import time
import openai
import argparse

message = [{'role':'system', 'content': 'You are a helpful assistant to figure out the scene based on the given actions.'}]

demos =[
       "take fork, wash fork, put fork, take knife, wash knife, take spoon, wash spoon, put spoon => clean tableware ###\n",
       
       "remove pencil, put pencil, take pencil, put pencil, put pencil, take pencil, remove container, put container => writing ###\n",
       
       "inspect gauge, take gauge, operate gauge, inspect gauge, put plier, take paper, clean gauge, put paper => fix car ###\n",
       
       "clean flour, put cutter, put pot, take container, put container, close cabinet, move pot, remove tray => baking ###\n",
       
       "clean floor, adjust table, put table, pull bucket, dip bucket, squeeze bucket, unfold napkin, clean napkin => set dining-table ###\n",
       
       "move wheel, take chain, repair chain, remove chain, remove chain, put chain, put bicycle, put bicycle => fix bicycle ###\n",
       
       "put plant, put plant, take plant, cut plant, put plant, take trowel, scoop bag, pour soil => gardening ###\n",
       
       "lift container, hit container, put container, shake container, pour soil, take plant, put bucket, put soil => planting ###\n",
       
       "take solder-iron, put solder-iron, hold wire, take solder-iron, attach wire, attach wire, put solder-iron, take rubber-band => solder wires ###\n",
       
       "put iron, adjust shirt, take iron, press shirt, put iron, adjust shirt, take shirt, put cloth => iron clothes ###\n",
       
       "take flower, take flower, take flower, take flower, take flower, take flower, take flower, cut flower => arrange flowers ###\n",
       
       "take broom, move broom, take dustpan, put dustpan, move broom, take dustpan, clean trash, put broom => clean room ###\n",

]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_dir', type=str, default="dataset", help="path to the output directory")
    parser.add_argument('--dataset_dir', type=str, default="dataset", help="path to the dataset directory")
    parser.add_argument('--val_name', type=str, default="val_nseg8_recog_subset600.jsonl", help="name of the validation file")
    parser.add_argument('--response_name', type=str, default="goal_val600.json", help="name of the output file")
    parser.add_argument('--openai_key', type=str, required=True, help="openai key")
    parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo', help="GPT api name")
    parser.add_argument('--temperature', type=float, default=0.3, help="GPT api parameter temperature")
    parser.add_argument('--n', type=int, default=1, help="GPT api parameter n")
    parser.add_argument('--max_tokens', type=int, default=500, help="GPT api parameter max_tokens")
    args = parser.parse_args()

    response_path = os.path.join(args.response_dir, args.response_name)
    val_path = os.path.join(args.dataset_dir, args.val_name)
    print('validation data path: ', val_path)
    print('Response saving path: ', response_path)

    val_df = pd.read_json(val_path, lines=True)
    val_x = val_df['prompt'].apply(lambda x: x.replace("\n","").replace("#","")[:-1]).tolist()
    val_y = val_df['completion'].apply(lambda x: x.strip().replace("\n","").replace("#","")[:-1]).tolist()

    openai.api_key = args.openai_key
    print('OpenAI key ending with: ', openai.api_key[-5:])

    val_idx = np.arange(len(val_x)).tolist()
    total_num = len(val_idx)
    over = False
    print("start ICL querying from {}".format(args.gpt_model))
    print('GPT api parameters: ', args.temperature, args.n, args.max_tokens)
    while not over:
        try:
            try:
                responses_list = json.load(open("val_subset600_scene.json", "r"))
            except:
                responses_list = []
                json.dump(responses_list, open(response_path, "w"))
            print("processed sample num: ", len(responses_list)) 

            for ii, prompt_idx in enumerate(val_idx):
                if ii < len(responses_list):
                    continue
                prompt_message = 'Suppose a person has performed the given actions in the form of a sequence of action pairs. Each action pair is defined by a {verb} and a {noun}, separated by a space. What is the most possible scene according to the given previous actions? Answer the scene using no more than three words. Here are some demonstrations and the questions: ###\n'
                for demo in demos:
                    prompt_message += demo
                prompt_message += val_x[prompt_idx] + " => "
                mes = message + [{'role':'user', 'content': prompt_message}]
                response = openai.ChatCompletion.create(
                                                model=args.gpt_model,
                                                messages=mes,
                                                max_tokens = args.max_tokens,
                                                n = args.n,
                                                temperature = args.temperature,
                                                )

                print(str(ii+1)+'/'+str(total_num) + ": ", val_x[ii])
                choices = response['choices']
                answer = choices[0]["message"]["content"].strip()
                responses_list.append(answer)
                json.dump(responses_list, open(response_path, "w"))
                print(answer)
                print("------------------------------------\n")
            over = True
        except Exception as e:
            print(e)
            print("Rate limit error, sleep for 30 seconds")
            time.sleep(30)