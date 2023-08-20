import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision
import copy
import pdb
import itertools
import random
from collections import defaultdict

from ..utils import file_util
from ..parser import load_config, parse_args


class BaseVideoDataset(Dataset):
    def __init__(self, cfg, annotation_path, is_train, is_test) -> None:
        super().__init__()
        self.cfg = cfg
        self.annotation_path = annotation_path
        if len(self.cfg.data.examples_to_keep) > 0:
            if is_train:
                self.examples_to_keep = set(file_util.load_json(self.cfg.data.examples_to_keep))  # set of last_observed_id 
            else:
                self.examples_to_keep = []
        else:
            self.examples_to_keep = []
        self.is_train = is_train
        self.is_test = is_test
        self.annotations = self.convert(file_util.load_json(self.annotation_path))

    def get_future_anno(self, segments, idx):
        
        [start, end] = self.cfg.data.output_segments
        if idx + start >= len(segments) or idx + end <= 0:
            # no label at all
            return None
        if idx + start < 0 or idx + end > len(segments):
            # need to be partially masked
            if not self.cfg.data.output_mask:
                return None
        output = {
            "verb": [-1] * (end - start),
            "noun": [-1] * (end - start),
            "mask": [True] * (end - start),
            "last_observed_id": "{}_{}".format(segments[idx-1]['clip_uid'], segments[idx-1]['action_idx']),
        }
        for i in range(idx+start, idx+end):
            if i < 0 or i >= len(segments):
                continue
            if not self.is_test:
                output['verb'][i-start-idx] = segments[i]['verb_label']
                output['noun'][i-start-idx] = segments[i]['noun_label']
            output['mask'][i-start-idx] = False
        return output

    def get_text_feature_anno(self, segments, idx):
        [start, end] = self.cfg.data.output_segments
        if idx + start >= len(segments) or idx + end <= 0:
            # no label at all
            return None
        if idx + start < 0 or idx + end > len(segments):
            # need to be partially masked
            if not self.cfg.data.output_mask:
                return None
        
        anno = {
            'key': segments[idx]['clip_uid']+"_"+str(segments[idx-1]['action_idx']),
            "mask": [False]
        }

        if anno['key'] in self.text_features.keys():
            return anno
        
    
    def get_text_anno(self, segments, idx):
        [start, end] = self.cfg.data.input_segments
        num_segments = end - start
        anno = {
            "verb": [-1] * num_segments,  # verb idx
            "noun": [-1] * num_segments,  # noun idx
            "mask": [True] * num_segments,  # True: masked
        }
        if self.cfg.data.input_from_annotated_segments:
            if idx + start >= len(segments) or idx + end <= 0:
                # no label at all
                return None
            if idx + start < 0 or idx + end > len(segments):
                # need to be partially masked
                if not self.cfg.data.input_mask:
                    return None
            for i in range(idx + start, idx + end):
                if i < 0 or i >= len(segments):
                    continue
                
                if self.is_train:
                    anno["verb"][i-idx-start] = segments[i]['verb_label']
                    anno["noun"][i-idx-start] = segments[i]['noun_label']
                else:
                    anno["verb"][i-idx-start] = self.predictions[segments[0]['clip_uid']+"_"+str(i)]['verb'][0][0]
                    anno["noun"][i-idx-start] = self.predictions[segments[0]['clip_uid']+"_"+str(i)]['noun'][0][0]
                    #raise NotImplementedError("Test set has no ground truth inputs.")
                anno["mask"][i-idx-start] = False
        else:
            raise NotImplementedError("Text modality only supports labelled segments as inputs.")
        return anno
        
    def get_image_anno(self, segments, idx):
        [start, end] = self.cfg.data.image.input_segments
        image_fps = self.cfg.data.image.fps
        strict = self.cfg.data.strict
        anno = {
            "path": '{}/{}.pt'.format(self.cfg.data.image.base_path, segments[idx]['clip_uid']),
            "verb": [],  # verb idx
            "noun": [],  # noun idx
            "meta_data": [],  # each element: frame index when extracting image features
            "mask": [],  # True: masked
        }
        if self.cfg.data.image.input_from_annotated_segments:
            if idx + start >= len(segments) or idx + end <= 0:
                # no label at all
                return None
            if idx + start < 0 or idx + end > len(segments):
                # need to be partially masked
                if not self.cfg.data.image.input_mask:
                    return None
            num_images_per_segment = self.cfg.data.image.num_images_per_segment
            for i in range(idx + start, idx + end):
                if i < 0 or i >= len(segments):
                    anno["verb"].extend([-1] * num_images_per_segment)
                    anno["noun"].extend([-1] * num_images_per_segment)
                    anno["meta_data"].extend([0] * num_images_per_segment)
                    anno["mask"].extend([True] * num_images_per_segment)
                    continue

                if self.is_test:
                    anno["verb"].extend([-1] * num_images_per_segment)
                    anno["noun"].extend([-1] * num_images_per_segment)
                else:
                    anno["verb"].extend([segments[i]['verb_label']] * num_images_per_segment)
                    anno["noun"].extend([segments[i]['noun_label']] * num_images_per_segment)

                segment_start_sec = segments[i]['action_clip_start_sec']
                segment_end_sec = segments[i]['action_clip_end_sec']
                if i == idx + end - 1 and strict:
                    segment_end_sec -= 1 / (image_fps * 2)
                    assert segment_end_sec > segment_start_sec, 'segment too short, consider turning off strict'

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
                if self.cfg.data.image.segment_feature:
                    anno["meta_data"].extend([i])
                else:
                    anno["meta_data"].extend(frame_indices)
                anno['mask'].extend([False] * len(frame_indices))

        else:
            num_images = self.cfg.data.image.num_images_per_segment
            interval = self.cfg.data.image.image_interval
            if self.cfg.data.image.from_end:
                end_sec = segments[idx]['action_clip_end_sec'] - self.cfg.data.tau_a
            else:
                end_sec = segments[idx]['action_clip_start_sec'] - self.cfg.data.tau_a
            if strict:
                end_sec -= 1 / (image_fps * 2)
            start_sec = end_sec - (num_images - 1) * interval
            if end_sec < 0:
                # no inputs at all
                return None
            if start_sec < 0:
                # need to be partially masked
                if not self.cfg.data.image.input_mask:
                    return None
            anno = {
                "path": '{}/{}.pt'.format(self.cfg.data.image.base_path, segments[idx]['clip_uid']),
                "verb": [-1] * num_images,  # verb idx
                "noun": [-1] * num_images,  # noun idx
                "meta_data": [0] * num_images,  # each element: frame index when extracting image features
                "mask": [True] * num_images,  # True: masked
            }

            def check_anno(t):
                for i in range(len(segments)):
                    if t < segments[i]['action_clip_end_sec']:
                        if i == len(segments) - 1: 
                            return -1
                        elif i+1 < idx:
                            return segments[i+1]['verb_label']
                        else:
                            return -1
             
            t = end_sec
            timesteps = []
            for i in range(num_images):
                if t < 0:
                    timesteps.append(0)
                else:
                    timesteps.append(t)
                    anno['mask'][i] = False
                    anno['verb'][num_images - i - 1] = check_anno(t)
                t -= interval
            timesteps.sort()
            anno['meta_data'] = [int(t * image_fps) for t in timesteps]

        return anno

    def convert(self, row_annotations):
        # get modalities
        modalities = ['future']
        if self.cfg.data.use_gt_text:
            modalities.append('text')
            self.predictions = file_util.load_json(self.cfg.data.prediction_path)  
        if self.cfg.data.use_goal:
            modalities.append('text_feature')
            if self.is_train:
                self.text_features = file_util.load_pickle(self.cfg.data.train_text_feature_path)
            else:
                self.text_features = file_util.load_pickle(self.cfg.data.val_text_feature_path)
                print(len(self.text_features))
        if self.cfg.model.img_feat_size > 0:
            modalities.append('image')


        segments_all = row_annotations['clips']
        annotations = []
        for _, group in itertools.groupby(segments_all, key=lambda x: x['clip_uid']):
            segment_info = sorted(list(group), key=lambda x: x['action_idx'])
            for i in range(len(segment_info)):
                anno = {}   # {modality_name: anno}
                for modality in modalities:
                    get_anno_func = getattr(self, f'get_{modality}_anno')
                    anno_single = get_anno_func(segment_info, i)
                    if anno_single is not None:
                        anno[modality] = anno_single
                if 'future' in anno and len(anno) > 1:
                    # have labels and at least one modality as inputs
                    if self.cfg.data.use_goal and (not 'text_feature' in anno): continue
                    annotations.append(anno)
              
        # filter 
        if len(self.examples_to_keep) == 0:
            return annotations
        filtered_annotations = []
        for annotation in annotations:
            if annotation['future']['last_observed_id'] in self.examples_to_keep:
                filtered_annotations.append(annotation)
        return filtered_annotations
        
    def fill_future(self, anno):
        anno['mask'] = torch.tensor(anno['mask'], dtype=torch.bool)

    def fill_text_feature(self, anno):
        key = anno['key']
        anno['inputs'] = self.text_features[key]
        anno['mask'] = torch.tensor(anno['mask'], dtype=torch.bool)

    def fill_text(self, anno):
        anno['inputs'] = torch.tensor([anno['verb'], anno['noun']]).T
        anno['mask'] = torch.tensor(anno['mask'], dtype=torch.bool)
        
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
                image_features.append(embs[offset_list])
            image_features = torch.cat(image_features, dim=0)
        else:
            image_features = torch.load(anno['path'], map_location='cpu')  # (N, D)
            image_features = image_features[indices]
        anno['inputs'] = image_features
        anno['mask'] = torch.tensor(anno['mask'], dtype=torch.bool)

    def __getitem__(self, index):
        annotation = copy.deepcopy(self.annotations[index])
        for modality in annotation:
            fill_func = getattr(self, f'fill_{modality}')
            fill_func(annotation[modality])

        observed_labels_idx = None
        for modality in annotation:
            if modality != 'future' and modality != 'text_feature':
                observed_labels_idx = torch.tensor([annotation[modality]['verb'], annotation[modality]['noun']]).T
        label = torch.tensor([annotation['future']['verb'], annotation['future']['noun']]).T
        item = {
            'forecast_labels_idx': label,
            'observed_labels_idx': observed_labels_idx,
            'last_observed_id': annotation['future']['last_observed_id'],
        }
        if self.cfg.data.use_gt_text:
            item['text'] = annotation['text']['inputs']
            item['mask_text'] = annotation['text']['mask']
        if self.cfg.data.use_goal:
            item['text_feature'] = annotation['text_feature']['inputs'].unsqueeze(dim=0)
            item['mask_text_feature'] = annotation['text_feature']['mask']
        if self.cfg.model.img_feat_size > 0:
            item['image_features'] = annotation['image']['inputs']
            item['mask_image'] = annotation['image']['mask']
            item['image_labels'] = torch.tensor(annotation['image']['verb'])

        return item

    def __len__(self):
        return len(self.annotations)
    

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if not hasattr(self, 'train_set'):
                self.train_set = BaseVideoDataset(self.cfg, self.cfg.data.train_anno_path, True, False)
                print(len(self.train_set))
            if not hasattr(self, 'val_set'):
                self.val_set = BaseVideoDataset(self.cfg, self.cfg.data.val_anno_path, False, False)
                print(len(self.val_set))


        if stage == "test" or stage is None:
            self.test_set = BaseVideoDataset(self.cfg,self.cfg.data.test_anno_path, False, True)

    def train_dataloader(self):
        if not hasattr(self, 'train_loader'):
            num_gpus = self.cfg.num_gpus
            assert self.cfg.train.batch_size % num_gpus == 0
            batch_size = self.cfg.train.batch_size // num_gpus
            self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size=batch_size, num_workers=self.cfg.train.num_workers)
        return self.train_loader

    def val_dataloader(self):
        if not hasattr(self, 'val_loader'):
            num_gpus = self.cfg.num_gpus
            assert self.cfg.val.batch_size % num_gpus == 0
            batch_size = self.cfg.val.batch_size // num_gpus
            self.val_loader = DataLoader(self.val_set, shuffle=False, batch_size=batch_size, num_workers=self.cfg.val.num_workers)
        return self.val_loader

    def test_dataloader(self):
        if not hasattr(self, 'test_loader'):
            # num_gpus = self.cfg.num_gpus
            num_gpus = 1
            assert self.cfg.test.batch_size % num_gpus == 0
            batch_size = self.cfg.test.batch_size // num_gpus
            self.test_loader = DataLoader(self.test_set, shuffle=False, batch_size=batch_size, num_workers=self.cfg.test.num_workers, drop_last=False)
        return self.test_loader

    
    
def sanity_check():

    args = parse_args()
    cfg = load_config(args)
    dm = BaseDataModule(cfg)
    dm.setup(stage="fit")


if __name__ == '__main__':
    sanity_check()
