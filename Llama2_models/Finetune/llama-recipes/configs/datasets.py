# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"
    
@dataclass
class ego4d_v1:
    dataset: str = "ego4d_v1"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/train_nseg8.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/val_nseg8_recog.csv"
    
@dataclass
class ego4d_v1_aug:
    dataset: str = "ego4d_v1_aug"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/train_nseg8_aug.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/val_nseg8_recog.csv"

@dataclass
class ego4d_v1_recog:
    dataset: str = "ego4d_v1_recog"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/train_nseg8_recog.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/val_nseg8_recog.csv"    

@dataclass
class ego4d_v1_aug_egovlp:
    dataset: str = "ego4d_v1_aug_egovlp"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/train_nseg8_aug_egovlp.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/val_nseg8_recog_egovlp.csv"
      
@dataclass
class ego4d_v2:
    dataset: str = "ego4d_v2"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog.csv"

@dataclass
class ego4d_v2_shuffle:
    dataset: str = "ego4d_v2_shuffle"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8_shuffle.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog.csv"
    
@dataclass
class ego4d_v2_aug:
    dataset: str = "ego4d_v2_aug"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8_aug.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog.csv"

@dataclass
class ego4d_v2_aug_shuffle:
    dataset: str = "ego4d_v2_aug_shuffle"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8_aug_shuffle.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog.csv"
    
@dataclass
class ego4d_v2_recog:
    dataset: str = "ego4d_v2_recog"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8_recog.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog.csv"

@dataclass
class ego4d_v2_recog_egovlp:
    dataset: str = "ego4d_v2_recog_egovlp"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8_recog_egovlp.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog_egovlp.csv"

@dataclass
class ego4d_v2_aug_egovlp:
    dataset: str = "ego4d_v2_aug_egovlp"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8_aug_egovlp.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog_egovlp.csv"
    
@dataclass
class ego4d_v2_3in1:
    dataset: str = "ego4d_v2_3in1"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8_3in1.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog.csv"

@dataclass
class idx_ego4d_v1:
    dataset: str = "idx_ego4d_v1"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_train_nseg8.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_val_nseg8_recog.csv"
    
@dataclass
class idx_ego4d_v1_aug:
    dataset: str = "idx_ego4d_v1_aug"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_train_nseg8_aug.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_val_nseg8_recog.csv"

@dataclass
class idx_ego4d_v1_recog:
    dataset: str = "idx_ego4d_v1_recog"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_train_nseg8_recog.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_val_nseg8_recog.csv"    
    
@dataclass
class idx_ego4d_v2:
    dataset: str = "idx_ego4d_v2"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_v2_train_nseg8.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_v2_val_nseg8_recog.csv"
    
    
@dataclass
class idx_ego4d_v2_aug:
    dataset: str = "idx_ego4d_v2_aug"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_v2_train_nseg8_aug.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_v2_val_nseg8_recog.csv"

@dataclass
class idx_ego4d_v2_recog:
    dataset: str = "idx_ego4d_v2_recog"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_v2_train_nseg8_recog.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_v2_val_nseg8_recog.csv"

@dataclass
class idx_ego4d_v2_aug_egovlp:
    dataset: str = "idx_ego4d_v2_aug_egovlp"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_v2_train_nseg8_aug_egovlp.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_v2_val_nseg8_recog.csv"
    
@dataclass
class idx_ego4d_v2_3in1:
    dataset: str = "idx_ego4d_v2_3in1"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_v2_train_nseg8_3in1.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/idx_v2_val_nseg8_recog.csv"

@dataclass
class vis_ego4d_v1:
    dataset: str = "vis_ego4d_v1"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/train_nseg8.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/val_nseg8_recog.csv"
    
@dataclass
class vis_ego4d_v1_aug:
    dataset: str = "vis_ego4d_v1_aug"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/train_nseg8_aug.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/val_nseg8_recog.csv"


@dataclass
class vis_ego4d_v1_recog:
    dataset: str = "vis_ego4d_v1_recog"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/train_nseg8_recog.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/val_nseg8_recog.csv"

@dataclass
class vis_ego4d_v2:
    dataset: str = "vis_ego4d_v2"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog.csv"
    
@dataclass
class vis_ego4d_v2_aug:
    dataset: str = "vis_ego4d_v2_aug"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8_aug.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog.csv"


@dataclass
class vis_ego4d_v2_recog:
    dataset: str = "vis_ego4d_v2_recog"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8_recog.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog.csv"

@dataclass
class vis_ego4d_v2_aug_caption:
    dataset: str = "vis_ego4d_v2_aug_caption"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8_aug_caption.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog_caption.csv"

@dataclass
class vis_ego4d_v2_aug_caption_only:
    dataset: str = "vis_ego4d_v2_aug_caption_only"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8_aug_caption_only.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8_recog_caption_only.csv"
    
@dataclass
class proj_ego4d_v1:
    dataset: str = "proj_ego4d_v1"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/train_nseg8.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/val_nseg8.csv"
    
@dataclass
class proj_ego4d_v2:
    dataset: str = "proj_ego4d_v2"
    train_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_train_nseg8.csv"
    test_split: str = "/users/swang299/code/AntGPT-Llama2/dataset/v2_val_nseg8.csv"
    

@dataclass
class ek_dataset:
    dataset: str = "ek_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/ek_dataset/annotations/"
    verb_dictionary_path = "EPIC_verb_classes.csv"
    noun_dictionary_path = "EPIC_noun_classes.csv"
    many_shot_path = "EPIC_many_shot_verbs.csv"
    use_recognition = True
    recog_file ='/gpfs/data/csun45/qzhao25/GAZE/anticipation/lightning_logs/ek_recog/ek_rec/test/submit.json'
    num_labels = 125
    problem_type = 'multi_label_classification'
    max_segments = 60
    train_Ks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    val_Ks = [.25, .50, .75]
    gaze = False
    set_input = False

@dataclass
class ek_dataset_si:
    dataset: str = "ek_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/ek_dataset/annotations/"
    verb_dictionary_path = "EPIC_verb_classes.csv"
    noun_dictionary_path = "EPIC_noun_classes.csv"
    many_shot_path = "EPIC_many_shot_verbs.csv"
    use_recognition = True
    recog_file ='/gpfs/data/csun45/qzhao25/GAZE/anticipation/lightning_logs/ek_recog/ek_rec/test/submit.json'
    num_labels = 125
    problem_type = 'multi_label_classification'
    max_segments = 60
    train_Ks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    val_Ks = [.25, .50, .75]
    gaze = False
    set_input = True

@dataclass
class gaze_dataset:
    dataset: str = "ek_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/ek_dataset/gaze_annotations/"
    verb_dictionary_path = "action_list_t+v.csv"
    noun_dictionary_path = "action_list_t+v.csv"
    many_shot_path = "gtea_many_shot_verbs.csv"
    use_recognition = False
    recog_file ='/gpfs/data/csun45/czhan164/gaze/logits.pkl'
    num_labels = 19
    problem_type = 'multi_label_classification'
    max_segments = 40
    train_Ks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    val_Ks = [.25, .50, .75]
    gaze = True
    set_input = False

@dataclass
class gaze_dataset_si:
    dataset: str = "ek_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/ek_dataset/gaze_annotations/"
    verb_dictionary_path = "action_list_t+v.csv"
    noun_dictionary_path = "action_list_t+v.csv"
    many_shot_path = "gtea_many_shot_verbs.csv"
    use_recognition = False
    recog_file ='/gpfs/data/csun45/czhan164/gaze/logits.pkl'
    num_labels = 19
    problem_type = 'multi_label_classification'
    max_segments = 40
    train_Ks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    val_Ks = [.25, .50, .75]
    gaze = True
    set_input = True
