import os
import copy
import itertools
import random
from collections import defaultdict
from pprint import pprint
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
import pytorch_lightning as pl
from ..utils import file_util
from ..parser import parse_args, load_config


class GazeLTADataset(Dataset):
    def __init__(self, cfg, annotation_path, is_train, is_val) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.is_val = is_val
        self.annotation_path = annotation_path
        self.annotations = self.convert(file_util.load_json(self.annotation_path))

    def get_future_anno(self, segments):
        df = pd.DataFrame(segments)
        l_output = []
        if self.is_train:
            target_Ks = self.cfg.multicls.train_Ks
        if self.is_val:
            target_Ks = self.cfg.multicls.val_Ks
        for i in target_Ks:
            hist_clips = df[df['elocation']<=i]
            pred_clips = df[df['slocation']>i]
            if len(hist_clips) < 3 or len(pred_clips) < 3:
                continue
            output = {
                'verb': [False] * self.cfg.model.num_classes[0],  # length: num_verb_classes
                "last_observed_id": "{}_{}".format(segments[hist_clips.index.values[-1]]['clip_uid'], segments[hist_clips.index.values[-1]]['action_idx']),
            }
            verbs = set(pred_clips['verb_label'])
            for i in set(verbs):
                output['verb'][i] = True
            l_output.append(output)
        return l_output

    def get_image_anno(self, segments):
        df = pd.DataFrame(segments)
        l_output = []
        if self.is_train:
            target_Ks = self.cfg.multicls.train_Ks
        if self.is_val:
            target_Ks = self.cfg.multicls.val_Ks
        for i in target_Ks:
            hist_clips = df[df['elocation']<=i]
            pred_clips = df[df['slocation']>i]
            if len(hist_clips) < 3 or len(pred_clips) < 3:
                continue
            image_fps = self.cfg.data.image.fps
            anno = {
                "path": '{}/{}.pt'.format(self.cfg.data.image.base_path, segments[0]['clip_uid']),
                "verb": [],  # verb idx
                "K": segments[0]['K'],
                "meta_data": [],  # each element: frame index when extracting image features
            }
                
            num_images_per_segment = self.cfg.data.image.num_images_per_segment
            
            for index,value in hist_clips.iterrows():
                anno["verb"].append(hist_clips.loc[index]['verb_label'])
                segment_start_sec = hist_clips.loc[index]['action_clip_start_sec']
                segment_end_sec = hist_clips.loc[index]['action_clip_end_sec']                    

                intervals = []  # list of [interval_start, interval_end]
                interval_duration = (segment_end_sec - segment_start_sec) / num_images_per_segment
                interval_start, interval_end = segment_start_sec, segment_start_sec
                for _ in range(num_images_per_segment):
                    interval_start = interval_end
                    interval_end = min(segment_end_sec, interval_end + interval_duration)
                    intervals.append([interval_start * image_fps, interval_end * image_fps])
                
                # from each interval, we randomly sample 1 image
                frame_indices = []
                for interval in intervals:
                    if self.is_train:
                        frame_idx = int(interval[0] + random.random() * (interval[1] - interval[0]))
                    else:
                        frame_idx = int(interval[0] + 0.5 * (interval[1] - interval[0]))
                    frame_indices.append(frame_idx)
                anno["meta_data"].extend(frame_indices)

                if self.cfg.multicls.max_num_segments > 0:
                    anno['verb'] = anno['verb'][-self.cfg.multicls.max_num_segments:]
                    anno['meta_data'] = anno['meta_data'][-self.cfg.multicls.max_num_segments*num_images_per_segment:]

            l_output.append(anno)

        return l_output

    def get_text_feature_anno(self, segments):
        df = pd.DataFrame(segments)
        l_output = []
        if self.is_train:
            target_Ks = self.cfg.multicls.train_Ks
        if self.is_val:
            target_Ks = self.cfg.multicls.val_Ks
        for i in target_Ks:
            hist_clips = df[df['elocation']<=i]
            pred_clips = df[df['slocation']>i]
            if len(hist_clips) < 3 or len(pred_clips) < 3:
                continue
            l_output.append(dict({'key': segments[0]['clip_uid']}))
        return l_output
    

    def convert(self, row_annotations):
        
        modalities = ['future']
        if self.cfg.model.img_feat_size > 0:
            modalities.append('image')
        if self.cfg.data.use_goal:
            modalities.append('text_feature')
            self.text_features = file_util.load_pickle(self.cfg.multicls.text_feature_path)
        
        segments_all = row_annotations['clips']
        segments_all.sort(key=lambda x: x['clip_uid'])  # for itertools.groupby
        annotations = []
        for _, group in itertools.groupby(segments_all, key=lambda x: x['clip_uid']):
            segment_info = sorted(list(group), key=lambda x: x['action_idx'])
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
        
        if self.is_val:
            eval_verbs = []
            for i in annotations:
                for j in range(len(i['future']['verb'])):
                    if i['future']['verb'][j] == 1: eval_verbs.append(j)
            eval_verbs = set(eval_verbs)
            label_mask = [True if i in eval_verbs else False for i in range(self.cfg.model.num_classes[0])]
            with open(self.cfg.multicls.many_shot_path, "r") as f:
                ls = f.readlines()[1:]
                freq = set([int(l.split(',')[0]) for l in ls])
            freq_verbs = eval_verbs & freq
            rare_verbs = eval_verbs - freq
            label_mask_freq = [True if i in freq_verbs else False for i in range(self.cfg.model.num_classes[0])]
            label_mask_rare = [True if i in rare_verbs else False for i in range(self.cfg.model.num_classes[0])]
            self.label_mask = torch.tensor([label_mask,label_mask_freq,label_mask_rare]).to(dtype=torch.bool)
        else:
            self.label_mask = None
   
        return annotations
    
    def fill_future(self, anno):
        pass

    def fill_image(self, anno):
        indices = anno['meta_data']
        num_frames_per_file = self.cfg.data.image.split 
        if num_frames_per_file > 0:
            file_id_and_offsets = defaultdict(list)  # {file_id: [offsets]}
            for frame_index in indices:
                file_id_and_offsets[frame_index // num_frames_per_file].append(frame_index % num_frames_per_file)
            file_id_and_offsets = sorted(list(file_id_and_offsets.items()), key=lambda x: x[0])  # [(file_id, [offsets])]
            image_features = []
            for file_id, offset_list in file_id_and_offsets:
                emb_fp = '{}-{}.pt'.format(anno['path'][:-3], file_id)
                embs = torch.load(emb_fp, map_location='cpu')  # (N, D)
                for i in range(len(offset_list)):
                    if offset_list[i] > embs.shape[0] - 1:
                        offset_list[i] = embs.shape[0] - 1
                image_features.append(embs[offset_list])
            image_features = torch.cat(image_features, dim=0)
        else:
            image_features = torch.load(anno['path'], map_location='cpu')  # (N, D)
            image_features = image_features[indices]
        anno['inputs'] = image_features
    
    def fill_text_feature(self, anno):
        key = anno['key']
        anno['inputs'] = self.text_features[key]

    def __getitem__(self, index):
        annotation = copy.deepcopy(self.annotations[index])
        for modality in annotation:
            fill_func = getattr(self, f'fill_{modality}')
            fill_func(annotation[modality])
        observed_labels_idx = None
        for modality in annotation:
            if modality != 'future':
                observed_labels_idx = torch.tensor(annotation[modality]['verb'])
                Ks = torch.tensor(annotation[modality]['K'])
                break
        item = {
            'forecast_labels_idx': torch.tensor(annotation['future']['verb']),  # (num_verb_classes, )
            'observed_labels_idx': observed_labels_idx,   # (num_input_segments, )
            'last_observed_id': annotation['future']['last_observed_id'],  # str
            'label_mask': self.label_mask,
        }
        if self.cfg.model.img_feat_size > 0:
            item['image_features'] = annotation['image']['inputs']  # (num_input_segments * num_images_per_segment, D)
        if self.cfg.data.use_goal:
            item['text_feature'] = annotation['text_feature']
        return item

    def __len__(self):
        return len(self.annotations)


class Collater(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, batch):
        max_num_segments = max([len(el['observed_labels_idx']) for el in batch])
        keys = list(batch[0].keys())
        output = {}
        for key in keys:
            if key == 'last_observed_id':
                output[key] = [el[key] for el in batch]
            elif key == 'forecast_labels_idx':
                output[key] = torch.stack([el[key] for el in batch], dim=0)
            elif key == 'image_features':
                m = max_num_segments * self.cfg.data.image.num_images_per_segment
                ls = []
                masks = []
                for el in batch:
                    t = el[key]  # (N, ...)
                    mask = [False] * m
                    for i in range(m-len(t)):
                        mask[i] = True
                    # (N, ...) --> (M, ...)
                    if m > len(t):
                        pad = torch.zeros((m - len(t), ) + (t[0].shape))
                        ls.append(torch.cat([pad, t], dim=0))  # pad
                    else:
                        ls.append(t) 
                    masks.append(torch.tensor(mask))
                output[key] = torch.stack(ls, dim=0)
                output['mask_image'] = torch.stack(masks, dim=0)
            elif key == 'text_feature':
                output[key] = torch.stack([el[key]['inputs'] for el in batch], dim=0).unsqueeze(dim=1)
                output['mask_text_feature'] = torch.stack([torch.tensor([False]) for el in batch], dim=0)
            elif key == 'label_mask':
                output[key] = batch[0][key]
                pass
            else:
                m = max_num_segments
                ls = []
                for el in batch:
                    t = el[key]  # (N, ...)
                    if m > len(t):
                        pad = torch.zeros((m - len(t), ) + (t[0].shape))
                        ls.append(torch.cat([pad, t], dim=0))  # pad
                    else:
                        ls.append(t) 
                output[key] = torch.stack(ls, dim=0)
        return output
    

class GazeLTADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.collater = Collater(cfg)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if not hasattr(self, 'train_set'):
                self.train_set = GazeLTADataset(self.cfg, self.cfg.data.train_anno_path, True, False)
                print(len(self.train_set))
            if not hasattr(self, 'val_set'):
                self.val_set = GazeLTADataset(self.cfg, self.cfg.data.val_anno_path, False, True)
                print(len(self.val_set))

    def train_dataloader(self):
        if not hasattr(self, 'train_loader'):
            num_gpus = self.cfg.num_gpus
            assert self.cfg.train.batch_size % num_gpus == 0
            batch_size = self.cfg.train.batch_size // num_gpus
            self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size=batch_size, num_workers=self.cfg.train.num_workers, drop_last=False, collate_fn=self.collater)
        return self.train_loader

    def val_dataloader(self):
        if not hasattr(self, 'val_loader'):
            num_gpus = self.cfg.num_gpus
            assert self.cfg.val.batch_size % num_gpus == 0
            batch_size = self.cfg.val.batch_size // num_gpus
            self.val_loader = DataLoader(self.val_set, shuffle=False, batch_size=batch_size, num_workers=self.cfg.val.num_workers, drop_last=False, collate_fn=self.collater)
        return self.val_loader


def sanity_check():
    args = parse_args()
    cfg = load_config(args)
    dm = GazeLTADataModule(cfg)
    dm.setup(stage="fit")
    pprint(len(dm.train_set.annotations))
    pprint(len(dm.val_set.annotations))
    print(dm.val_set.label_mask)
    for i in range(len(dm.val_set.annotations)):
        s = set()
        for j in range(len(dm.val_set.annotations[i]['future']['verb'])):
            if dm.val_set.annotations[i]['future']['verb'][j]: s.add(j)
        print(s)


if __name__ == '__main__':
    sanity_check()
    # main()

