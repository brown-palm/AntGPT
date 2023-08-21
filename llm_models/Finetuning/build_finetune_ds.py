import pandas as pd
from preprocessing import remove_word_synonyms

def build_finetune_dataset(datasets:list, save_jsonl=True):
    '''
    datasets: list of data files sampled from ego4d annotations. Usually just training and validation.
    '''
    ret = []
    for num in range(len(datasets)):
        texts=[] #list of prompt
        labels=[] #list of completion
        for m in datasets[num]:
            
            nseg = len(m['hist_verb'])  
            #remove synonyms
            hist_verb = [ remove_word_synonyms(i) for i in m['hist_verb']]
            hist_noun = [ remove_word_synonyms(i) for i in m['hist_noun']]
            pred_verb = [ remove_word_synonyms(i) for i in m['pred_verb']]
            pred_noun = [ remove_word_synonyms(i) for i in m['pred_noun']]

            #form a prompt
            text=''
            for k in [i+" "+j for i,j in zip(hist_verb,hist_noun)]:
                v,n=k.split(" ")
                text+=k+", "
            texts.append(text[:-2]+ " \n\n###\n\n") #a suffix that openai recommend

            #form a completion
            label=' '#open ai wants the completion to start with a blank space
            for k in [i+" "+j for i,j in zip(pred_verb,pred_noun)]:
                v,n=k.split(" ")
                label+=k+", "
            labels.append(label[:-2]+" ###") #a suffix in completion that openai recommend
            
        if num == 0:
            ret.append(pd.DataFrame(zip(texts, labels), columns = ['prompt','completion']))
            if save_jsonl:
                train.to_json(f"train_nseg{nseg}.jsonl", orient='records', lines=True)
        elif num == 1:
            ret.append(pd.DataFrame(zip(texts, labels), columns = ['prompt','completion']))
            if save_jsonl:
                val.to_json(f"val_nseg{nseg}.jsonl", orient='records', lines=True)
        else:
            raise NotImplementedError("More than two data file is not supported")    
    return ret   