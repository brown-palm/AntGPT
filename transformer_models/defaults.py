from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# This will create a folder named `exp_name` under 'lightning_logs' to save logs, models, etc.
_C.exp_name = ""

# rand seed
_C.seed = 1

# GPU
_C.num_gpus = 1

_C.pretrained_backbone_path = ""
# Must be pl checkpoint. This can override _C.pretrained_backbone_path.
_C.ckpt_path = ""


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.train = CfgNode()

# Enable training
_C.train.enable = False

# Could be 'ddp' or 'cpu'
_C.train.strategy = "ddp"

_C.train.limit_train_batches = 1.0
_C.train.limit_val_batches = 1.0

# save the best checkpoint by what metric?
_C.train.checkpoint_metric = ""
# "min", "max", "auto"
_C.train.checkpoint_mode = ""

# nume_workers per GPU
_C.train.num_workers = 8
# batchsize all GPU. Should be a multiple of num_gpus
_C.train.batch_size = 64


# ---------------------------------------------------------------------------- #
# Validation options.
# ---------------------------------------------------------------------------- #
_C.val = CfgNode()

_C.val.val_only = False
# nume_workers per GPU
_C.val.num_workers = 8
# batchsize all GPU. Should be a multiple of num_gpus
_C.val.batch_size = 64


# ---------------------------------------------------------------------------- #
# Test options.
# ---------------------------------------------------------------------------- #
_C.test = CfgNode()

# Enable training
_C.test.enable = False

# nume_workers per GPU
_C.test.num_workers = 32
# batchsize all GPU. Should be a multiple of num_gpus
_C.test.batch_size = 4

_C.test.limit_test_batches = 1.0

# generate logits
_C.test.gen_logits = False


# ---------------------------------------------------------------------------- #
# Solver options.
# ---------------------------------------------------------------------------- #
_C.solver = CfgNode()

_C.solver.num_epochs = 40

_C.solver.lr = 2e-4

_C.solver.weight_decay = 0.0
# learning rate policy
_C.solver.lr_policy = 'cosine_warmup'

_C.solver.warmup_epochs = 3

# optimizer
_C.solver.optimizer = "sgd"

# for SGD
_C.solver.momentum = 0.9

_C.solver.nesterov = True


# ---------------------------------------------------------------------------- #
# Model options.
# ---------------------------------------------------------------------------- #
_C.model = CfgNode()

# how many classes to predict
_C.model.num_classes = [115, 478]
# how many future queries to predict
_C.model.num_actions_to_predict = 1
# must be data.output_segments[1] - data.output_segments[0]
_C.model.total_actions_to_predict = 1
# how many sequences to sample from prediction
_C.model.num_sequences_to_predict = 5
# sampling method for predicted sequence: naive, action_sample, action_max
_C.model.sampleing_method = ['naive']
# claasification. Possibly segmentation, detection in the future
_C.model.model = "classification"
# pte, trf
_C.model.aggregator = "pte"
# mlp, multihead
_C.model.decoder = "mlp"
# ltaweightedloss,
_C.model.loss_fn = "LTAWeightedLoss"
# e.g. [0.5, 0.5] for verb and noun
_C.model.loss_wts_heads = [0.5, 0.5]
# e.g. [1/20]*20: equal weights for each step
_C.model.loss_wts_temporal = [1/20]*20
# -1: no image feature
_C.model.img_feat_size = -1
# Image features and object features will be projected to this size.
_C.model.base_feat_size = 2048
# whether to use gt text
_C.model.text_feat_size = -1
# DropToken for image embeddings
_C.model.drop_img = 0.0
# DropToken for text embeddings
_C.model.drop_text = 0.0
# PTE
_C.model.pte = CfgNode()

_C.model.pte.num_heads = 8

_C.model.pte.num_layers = 3

_C.model.pte.enc_dropout = 0.1

_C.model.pte.pos_dropout = 0.1
#decoding options
_C.model.autoregressive = False
#whether use teacherforcing in training, only matters when autoregressive
_C.model.teacherforcing = False


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.data = CfgNode()
# anticipation time. in seconds. 
_C.data.tau_a = 0.0

_C.data.strict = True
# path to annotation path
_C.data.train_anno_path = ""
_C.data.val_anno_path = ""
_C.data.test_anno_path = ""
# examples to keep (file path).  "" keeps everything.
_C.data.examples_to_keep = ""
# path to video data, e.g. 'data/ego4d/clips'
_C.data.base_path = ""
# e.g. ".mp4"
_C.data.suffix = ''

_C.data.vocab_path = None

# [start, end). e.g. [-3, 10).  0 is the current segment.
_C.data.output_segments = [0, 20]
# If False, drop examples that do not have enough outputs.
_C.data.output_mask = False
# [start, end). e.g. [-3, 10).  0 is the current segment.
_C.data.input_segments = [-3, 0]
# If False, drop examples without enough input segments. 
# If True, perform segment-level mask
_C.data.input_mask = False
# If True, use annotated segments; else, construct segments using segment_length and segment_interval.
_C.data.input_from_annotated_segments = True
# Used when input_from_annotated_segments=False. in seconds
_C.data.segment_length = 0
# Used when input_from_annotated_segments=False. in seconds
_C.data.segment_interval = 0
# 'random', 'last', 'multi_uniform'
_C.data.clip_sampler = 'random'
# used when clip_sampler = multi_uniform
_C.data.num_clips_per_segment = 1
# in seconds
_C.data.clip_length = 2.0
# The mean value of the video raw pixels across the R G B channels.
_C.data.mean = [0.45, 0.45, 0.45]
# The std value of the video raw pixels across the R G B channels.
_C.data.std = [0.225, 0.225, 0.225]
# The spatial crop size of the input clip.
_C.data.crop_size = 224
# If True, perform random horizontal flip on the video frames during training.
_C.data.random_flip = True
_C.data.random_flip_rate = 0.5
# The spatial augmentation jitter scales for training.
_C.data.jitter_scales = [256, 320]
# The number of frames of the input clip.
_C.data.num_frames = 32


# Image Data Config
_C.data.image = CfgNode()
_C.data.image.base_path = 'data/ego4d/image_features'
# how many frames to split when generateing features
_C.data.image.split = -1
# fps used when extracting image features
_C.data.image.fps = -1.0
# If True, use annotated segments; else, use image directly (AVT style).
_C.data.image.input_from_annotated_segments = True
# Used when input_from_annotated_segments = True. [start, end). e.g. [-3, 10).  0 is the current segment.
_C.data.image.input_segments = [-3, 0]
# If input_from_annotated_segments = True: Uniformly sample images from segments.
# If input_from_annotated_segments = False: Uniformly sample images before the start or end (data.image.from_end) of the current segment.
_C.data.image.num_images_per_segment = 4
# Used when input_from_annotated_segments = False. Sample images starting from the end of the current segment.
_C.data.image.from_end = False
# Used when input_from_annotated_segments = False. in seconds. Need to consider data.image.fps.
_C.data.image.image_interval = 1

_C.data.image.segment_feature = False
# If False, drop examples without enough input segments or images. 
_C.data.image.input_mask = False
# whether to use gt text
_C.data.use_gt_text = False

_C.data.prediction_path = ''
# whether to use goal
_C.data.use_goal = False

_C.data.train_text_feature_path = ''
_C.data.val_text_feature_path = ''

# -----------------------------------------------------------------------------
# EK/Gaze specific options
# -----------------------------------------------------------------------------
_C.multicls = CfgNode()

_C.multicls.enable = False

_C.multicls.many_shot_path = 'data/gaze/annotations/gtea_many_shot_verbs.csv'

_C.multicls.train_Ks = []

_C.multicls.val_Ks = []

_C.multicls.max_num_segments = -1

_C.multicls.text_feature_path = 'text_features/feature_gt.pkl'


def get_cfg():
    return _C.clone()