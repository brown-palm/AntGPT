import copy
import json
import pickle
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import List
import itertools

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class EKDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words = 200):
        super().__init__()
        self.dataset_config = dataset_config
        self.max_words = max_words
        self.tokenizer = tokenizer
        
        self.base_path = self.dataset_config.data_path
        self.recognition_result = None
        if partition == "train":
            self.ann = os.path.join(self.base_path,"train.json")
            self.target_Ks = self.dataset_config.train_Ks
        else:
            self.ann = os.path.join(self.base_path,"val.json")
            self.target_Ks = self.dataset_config.val_Ks
            if dataset_config.use_recognition:
                if not self.dataset_config.gaze:
                    with open(dataset_config.recog_file, 'r') as f:
                        self.recognition_result = json.load(f)
                else:
                    with open(dataset_config.recog_file, 'rb') as f:
                        self.recognition_result = pickle.load(f)
            
        if not self.dataset_config.gaze:
            verb_d = pd.read_csv(os.path.join(self.base_path,self.dataset_config.verb_dictionary_path))[['verb_id','class_key']]
        else:
            verb_d = pd.read_csv(os.path.join(self.base_path,self.dataset_config.verb_dictionary_path))[['verb_idx','verb']]
        self.verb_dictionary = dict()
        for index, value in verb_d.iterrows():
            self.verb_dictionary.update({value[0]:value[1]})
        
        if not self.dataset_config.gaze:
            noun_d = pd.read_csv(os.path.join(self.base_path,self.dataset_config.noun_dictionary_path))[['noun_id','class_key']]
        else:
            noun_d = pd.read_csv(os.path.join(self.base_path,self.dataset_config.noun_dictionary_path))[['noun_idx','noun']]
        self.noun_dictionary = dict()
        for index, value in noun_d.iterrows():
            self.noun_dictionary.update({value[0]:value[1]})
    
        with open(self.ann, 'r') as json_file:
            self.ann = json.load(json_file)
        
        self.annotations = self.convert(self.ann)
        self.get_label_mask(partition)
        
    def get_future_anno(self, segments):
        
        df = pd.DataFrame(segments)
        l_output = []
        
        for i in self.target_Ks:
            hist_clips = df[df['elocation']<=i]
            pred_clips = df[df['slocation']>i]
            if len(hist_clips) < 3 or len(pred_clips) < 3:
                continue
                
            output = {
                'verb': [0.0] * self.dataset_config.num_labels,
                "last_observed_id": "{}_{}".format(segments[hist_clips.index.values[-1]]['clip_uid'], segments[hist_clips.index.values[-1]]['action_idx']),
            }
            verbs = set(pred_clips['verb_label'])
            for i in set(verbs):
                output['verb'][i] = 1.0
            l_output.append(output)
            
        return l_output
    
    def get_text_anno(self, segments):
        df = pd.DataFrame(segments)
        l_output = []
        for i in self.target_Ks:
            hist_clips = df[df['elocation']<=i]
            pred_clips = df[df['slocation']>i]
            if len(hist_clips) < 3 or len(pred_clips) < 3:
                continue
                
            lv = hist_clips['verb_label'][-self.dataset_config.max_segments:].values.tolist()
            ln = hist_clips['noun_label'][-self.dataset_config.max_segments:].values.tolist()
            if not self.dataset_config.set_input:
                s = ''
                for i,j in zip(lv,ln):
                    s += self.verb_dictionary[i]+" "+self.noun_dictionary[j] + ", "
                l_output.append(dict({'action': s[:-2]+" =>"}))
            else:
                action_set = set()
                for i,j in zip(lv,ln):
                    action = self.verb_dictionary[i]+" "+self.noun_dictionary[j] + ", "
                    action_set.add(action)
                s = ''
                for a in action_set:
                    s += a
                l_output.append(dict({'action': s[:-2]+" =>"}))
        return l_output
    

    def convert(self, row_annotations):
        
        modalities = ['future']
        modalities.append('text')
        
        segments_all = row_annotations['clips']
        segments_all.sort(key=lambda x: x['clip_uid'])
        annotations = []
        for _, group in itertools.groupby(segments_all, key=lambda x: x['clip_uid']):
            segment_info = sorted(list(group), key=lambda x: x['action_idx']) #each video
            if (self.recognition_result is not None) and self.dataset_config.use_recognition:
                if not self.dataset_config.gaze:
                    for i in segment_info:
                        key = i['clip_uid'] + "_" + str(i['action_idx'])
                        i['verb_label'] = self.recognition_result[key]['verb'][0][0]
                        i['noun_label'] = self.recognition_result[key]['noun'][0][0]
                else:
                    for i in segment_info:
                        key = i['clip_uid'] + "_" + str(i['action_idx'])
                        i['verb_label'] = self.recognition_result[key]['verb'].softmax(dim=-1).argmax().item()
                        i['noun_label'] = self.recognition_result[key]['noun'].softmax(dim=-1).argmax().item()
            anno = {}   # {modality_name: anno}
            for modality in modalities:
                get_anno_func = getattr(self, f'get_{modality}_anno')
                anno_single = get_anno_func(segment_info)
                if anno_single is not None:
                    anno[modality] = anno_single
            anno = [
                {key: anno[key][index] for key in anno} for index in range(len(anno['future']))
            ]
            annotations+=anno
        
        return annotations

    def get_label_mask(self, partition):
        if partition != 'train':
            eval_verbs = []
            for i in self.annotations:
                for j in range(len(i['future']['verb'])):
                    if i['future']['verb'][j] == 1: eval_verbs.append(j)
            eval_verbs = set(eval_verbs)
            label_mask = [True if i in eval_verbs else False for i in range(self.dataset_config.num_labels)]
            many_shot_path = os.path.join(self.base_path, self.dataset_config.many_shot_path)
            with open(many_shot_path, "r") as f:
                ls = f.readlines()[1:]
                freq = set([int(l.split(',')[0]) for l in ls])
            freq_verbs = eval_verbs & freq
            rare_verbs = eval_verbs - freq
            label_mask_freq = [True if i in freq_verbs else False for i in range(self.dataset_config.num_labels)]
            label_mask_rare = [True if i in rare_verbs else False for i in range(self.dataset_config.num_labels)]
            self.label_mask = torch.tensor([label_mask,label_mask_freq,label_mask_rare]).to(dtype=torch.bool)
        else:
            self.label_mask = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        prompt = ann["text"]["action"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )

        example = prompt
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        example_mask = example.ge(0)
        example[~example_mask] = 0
        example_mask = example_mask.float()

        return {
            "input_ids": example,
            "labels": ann['future']['verb'],
            "attention_mask":example_mask,
        }


def sanity_check():
    return

if __name__ == '__main__':
    sanity_check()
