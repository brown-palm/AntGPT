import json
import os
import pandas as pd
import numpy as np
import pickle
import argparse
import editdistance
from rapidfuzz import fuzz, process
import re
# get index to verb/noun and verb/noun to index dict
dicts = json.load(open("dataset/dicts.json","r"))
val_words = json.load(open("dataset/val_words_nseg8.json","r"))
index_to_verb = dicts['i2v']
verb_to_index = dicts['v2i']
noun_to_index = dicts['n2i']
index_to_noun = dicts['i2n'] 
print('index_to_verb len: ', len(index_to_verb))
print('index_to_noun len: ', len(index_to_noun))
print('verb_to_index len: ', len(verb_to_index))
print('noun_to_index len: ', len(noun_to_index))  
    
def w2i_input(prompt_list):
    verb_idx = []
    noun_idx = []

    for prompt in prompt_list:
        prompt = prompt.split(", ")
        v_list = []
        n_list = []
        for vn in prompt:
            v, n = vn.split()
            v_list.append(verb_to_index[v])
            n_list.append(noun_to_index[n])
        verb_idx.append(v_list)
        noun_idx.append(n_list)

    verb_idx = np.array(verb_idx)
    noun_idx = np.array(noun_idx)
    return verb_idx, noun_idx

def handle_length_issues(action_seq, idx, jdx):
    global long
    global short
    train_idx = np.argsort(similar_matrix[idx])[:-5][jdx]
    if len(action_seq) > 20:
        long += 1
        return action_seq[:20]
    elif len(action_seq) < 20:
        short += 1
        to_r = 20 - len(action_seq)
        if to_r == 20:
            return train_y[train_idx].strip().split(", ")
        else:
            train_y[train_idx].strip().split(", ")
            return action_seq + [action_seq[-1] for i in range(to_r)]
    else:
        return action_seq

def word_to_idx(word,prime_dict,bk_dict,score_cutoff=90):
    global find
    global call
    call+=1
    try:
        #try synonym
        ret = prime_dict[word]
        find += 1
        return ret
    except:
        try:
            #try nearest neighbor, in practice not more helpful than label as the class with top-prob
            word = process.extractOne(word, list(prime_dict.keys()),score_cutoff=score_cutoff)[0]
            ret = prime_dict[word]
            find += 1
            return ret
        except:
            #handle edge-cases, very little occasions
            #return 1000 #just treat as wrong
            if bk_dict:
                choice = list(bk_dict.keys())
                prob = np.array(list(bk_dict.values()))/np.sum(list(bk_dict.values()))
                word = np.random.choice(choice,p=prob)
                return prime_dict[word]
            else:
                return 0

def edit_distance(preds, labels):
    """
    Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein
    Lowest among K predictions
    """
    if len(preds.shape) == 2:
        N, Z = preds.shape
        dists = []
        for n in range(N):
            dist = editdistance.eval(preds[n], labels[n])/Z
            dists.append(dist)
        return np.mean(dists)
        
    elif len(preds.shape) == 3:
        N, Z, K = preds.shape
        dists = []
        for n in range(N):
            dist = min([editdistance.eval(preds[n, :, k], labels[n])/Z for k in range(K)])
            dists.append(dist)
        return np.mean(dists)

    elif len(preds.shape) == 1:
        Z = len(preds)
        dist = editdistance.eval(preds, labels)/Z
        return np.mean(dist)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_dir', type=str, default="output", help="path to the output directory")
    parser.add_argument('--dataset_dir', type=str, default="dataset", help="path to the dataset directory")
    parser.add_argument('--response_name', type=str, default="subset600_responses_icl_recog.json", help="name of the output file")
    parser.add_argument('--train_name', type=str, default="idx_train_nseg8.csv", help="name of the training file")
    parser.add_argument('--val_name', type=str, default="val_nseg8_recog.jsonl", help="name of the validation file")
    parser.add_argument('--similar_matrix_name', type=str, default="similar_matrix.pkl", help="name of the similar matrix file")
    args = parser.parse_args()

    response_path = os.path.join(args.response_dir, args.response_name)
    train_path = os.path.join(args.dataset_dir, args.train_name)
    val_path = os.path.join(args.dataset_dir, args.val_name)
    similar_matrix_path = os.path.join(args.dataset_dir, args.similar_matrix_name)
    
    print('response data path: ', response_path)
    print('training data path: ', train_path)
    print('validation data path: ', val_path)
    print('similar matrix path: ', similar_matrix_path)
    
    # train_df = pd.read_json(train_path, lines=True)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_json(val_path, lines=True)

    train_x = train_df['prompt'].apply(lambda x: x.replace("\n","").replace("#","")[:-1]).tolist()
    train_y = train_df['completion'].apply(lambda x: x.strip().replace("\n","").replace("#","")[:-1]).tolist()
    val_x = val_df['prompt'].apply(lambda x: x.replace("\n","").replace("#","")[:-1]).tolist()
    val_y = val_df['completion'].apply(lambda x: x.strip().replace("\n","").replace("#","")[:-1]).tolist()

    similar_matrix = pickle.load(open(similar_matrix_path,"rb"))

    # val_x_verb, val_x_noun  = w2i_input(val_x)
    # val_y_verb, val_y_noun = w2i_input(val_y)
    # train_x_verb, train_x_noun  = w2i_input(train_x)
    # train_y_verb, train_y_noun = w2i_input(train_y)
    
    responses_list = json.load(open(response_path, "r"))
    print('response length: ', len(responses_list))

    # parse responses
    answers_list = []
    answer_len_list = []
    for response in responses_list:
        answers = []
        answer_len = []
        for choice in response:
            try:
                answer = choice.strip().split(", ")
                answers.append(answer)            
                answer_len.append(len(answer))
            except:
                answers.append([])
                answer_len.append(0)
        answers_list.append(answers)
        answer_len_list.append(answer_len)
       
    # split answers into verb and noun and handle length issues
    long,short=0,0
    processed_answers_v = []
    processed_answers_n = []
    for idx, answer in enumerate(answers_list):
        cand_seqs_v = []
        cand_seqs_n = []
        for jdx, choice in enumerate(answer):
            actions = choice
            temp_action = []
            for a in actions:
                a = a.strip()
                try:
                    v,n = a.split()
                    temp_action.append(a)
                except:
                    words = a.split()
                    real_verb = ""
                    real_noun = ""
                    for word in words:
                        if word in index_to_verb.keys():
                            real_verb = word
                            break
                        else:
                            real_verb = "93"
                    for word in words:
                        if word in index_to_noun.keys():
                            real_noun = word
                            break
                        else:
                            real_noun = "132"
                    temp_action.append(real_verb + " " + real_noun)
            actions = handle_length_issues(temp_action, idx, jdx)
            assert len(actions) == 20
            seq_v = []
            seq_n = []
            for a in actions:
                # print(a)
                try:
                    v,n = a.split()
                    v = v.strip()
                    n = n.strip()
                except:
                    v,n = "93", "132"
                seq_v.append(v.strip('.'))
                seq_n.append(n.strip('.'))
            cand_seqs_v.append(seq_v)
            cand_seqs_n.append(seq_n)
        processed_answers_v.append(cand_seqs_v)
        processed_answers_n.append(cand_seqs_n)
    processed_answers_v=np.swapaxes(np.array(processed_answers_v),1,2)
    processed_answers_n=np.swapaxes(np.array(processed_answers_n),1,2)

    
    verbs_answers = processed_answers_v.astype(int)
    nouns_answers = processed_answers_n.astype(int)
    
    val_y_verb = np.stack(pd.DataFrame(val_words)['pred_verb_idx'].values)
    val_y_noun = np.stack(pd.DataFrame(val_words)['pred_noun_idx'].values)
    print('ed_verb: ', edit_distance(verbs_answers, val_y_verb))
    print('ed_noun: ', edit_distance(nouns_answers, val_y_noun))
    print('ed_action: ', edit_distance(verbs_answers+nouns_answers,val_y_verb+val_y_noun))