import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

def handle_length_issues(action_seq):

    if len(action_seq) > 20:
        return action_seq[:20]
    elif len(action_seq) < 20:
        to_r = 20 - len(action_seq)
        if to_r == 20:
            return ["No match" for jjj in range(20)] #treat them as entirely wrong sequence or query for response again
        return action_seq + [action_seq[-1] for i in range(to_r)]
    else:
        return action_seq
    
def clean_responses(answers):
    processed_answers_v = []
    processed_answers_n = []
    for num,i in enumerate(answers):
        cand_seqs_v = []
        cand_seqs_n = []
        for j in i:
            actions = j[1:].split(", ")
            temp_action = []
            for a in actions:
                try:
                    v,n = a.split()
                    temp_action.append(a)
                except:
                    pass
            actions = handle_length_issues(temp_action)
            assert len(actions) == 20
            seq_v = []
            seq_n = []
            for a in actions:
                v,n = a.split()
                seq_v.append(v)
                seq_n.append(n)
            cand_seqs_v.append(seq_v)
            cand_seqs_n.append(seq_n)
        processed_answers_v.append(cand_seqs_v)
        processed_answers_n.append(cand_seqs_n)
        
    processed_answers_v=np.swapaxes(np.array(processed_answers_v),1,2)
    processed_answers_n=np.swapaxes(np.array(processed_answers_n),1,2)
   
    return processed_answers_v,processed_answers_n

def word_to_idx(word,prime_dict,bk_dict=None,score_cutoff=90):
    '''
    word: a single word to map back to index
    prime_dict: the primary word2index dictionary
    bk_dict: if cannot find match word in prime_dict even w. nearest neighbor, then sample from this dict.
    '''
    try:
        #try synonym
        ret = prime_dict[word]
        return ret
    except:
        try:
            #try nearest neighbor, in practice not more helpful than label as the class with top-prob
            word = process.extractOne(word, list(prime_dict.keys()),score_cutoff=score_cutoff)[0]
            ret = prime_dict[word]
            return ret
        except:
            #handle edge-cases, very little occasions
            if bk_dict:
                choice = list(bk_dict.keys())
                prob = np.array(list(bk_dict.values()))/np.sum(list(bk_dict.values()))
                word = np.random.choice(choice,p=prob)
                return prime_dict[word]
            else:
                return 0

def map_responses_to_index(responses:list,prime_dicts:list,bk_dicts=[None,None]):
    
    processed_answers_v,processed_answers_n = responses
    verb_to_index,noun_to_index=prime_dicts
    bk_verb,bk_noun = bk_dicts
    
    #vectorization for faster speed
    vectorized_word_to_idx = np.vectorize(word_to_idx)
    verbs_answers = vectorized_word_to_idx(processed_answers_v,prime_dict=verb_to_index,bk_dict=bk_verb)
    nouns_answers = vectorized_word_to_idx(processed_answers_n,prime_dict=noun_to_index,bk_dict=bk_noun)
    return verbs_answers, nouns_answers

def get_label(data):
    lv = np.stack(pd.DataFrame(data)['pred_verb_idx'].values)
    ln = np.stack(pd.DataFrame(data)['pred_noun_idx'].values)
    return lv,ln