import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import torch.distributed as dist
import tensorboard

from .parser import load_config, parse_args
from .utils import logging
from .tasks import load_task
from .datasets.get_datamodule import get_dm
import sys
sys.path.append("..")

logger = logging.get_logger(__name__)


def cleanup():
    dist.destroy_process_group()


def train(args, cfg):
    seed_everything(cfg.seed)

    # setup logger
    log_path = '{}'.format(args.exp_name)

    # dataset module
    dm = get_dm(cfg)
    dm.setup('fit')
    dm.train_dataloader()  # initialize dm.train_loader
    steps_in_epoch = len(dm.train_loader) // cfg.num_gpus
    print('steps_in_epoch: ', steps_in_epoch)

    # task module
    task = load_task(cfg, steps_in_epoch)

    # trainer setting
    tb_logger = TensorBoardLogger(save_dir='.', name='lightning_logs', version=log_path)
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.train.checkpoint_metric,
        mode=cfg.train.checkpoint_mode,
        save_last=True,
        save_top_k=1
    )
    learning_rate_callback = LearningRateMonitor()
    trainer_args = {}
    if cfg.train.strategy == 'ddp':
        trainer_args = {
            'accelerator': 'gpu',
            'devices': cfg.num_gpus,   # devices=[1, 3] to specify GPUs
            'strategy': 'ddp',
            # 'strategy': DDPStrategy(find_unused_parameters=False, static_graph=True, gradient_as_bucket_view=True),
        }
    elif cfg.train.strategy == 'cpu':
        trainer_args = {}
    trainer = Trainer(
        max_epochs=cfg.solver.num_epochs,

        benchmark=True,

        fast_dev_run=args.fast_dev_run,
        limit_train_batches=cfg.train.limit_train_batches,  # to avoid tensorboard issue
        limit_val_batches=cfg.train.limit_val_batches,  # to avoid tensorboard issue

        logger=tb_logger,
        callbacks=[learning_rate_callback, checkpoint_callback],

        **trainer_args,
    )
    trainer.fit(model=task, datamodule=dm, ckpt_path=cfg.ckpt_path)
    cleanup()
    return checkpoint_callback.best_model_path


def test(args, cfg, ckpt_path):
    seed_everything(cfg.seed)

    # setup logger
    log_path = '{}/test'.format(args.exp_name)

    # dataset module
    dm = get_dm(cfg)

    # task module
    task = load_task(cfg)

    tb_logger = TensorBoardLogger(save_dir='.', name='lightning_logs', version=log_path)

    # devices=1 to avoid distributed sampler.
    trainer = Trainer(
        accelerator = 'gpu',
        logger = tb_logger,
        devices = 1,
        limit_test_batches = cfg.test.limit_test_batches,
    )
    trainer.test(model=task, datamodule=dm, ckpt_path=ckpt_path)


def val(args, cfg, ckpt_path):
    seed_everything(cfg.seed)

    # setup logger
    log_path = '{}/val'.format(args.exp_name)

    # dataset module
    dm = get_dm(cfg)
    dm.setup('fit')

    # task module
    task = load_task(cfg)

    tb_logger = TensorBoardLogger(save_dir='.', name='lightning_logs', version=log_path)

    trainer_args = {}
    if cfg.train.strategy == 'ddp':
        trainer_args = {
            'accelerator': 'gpu',
            'devices': cfg.num_gpus,   # devices=[1, 3] to specify GPUs
            'strategy': 'ddp',
            # 'strategy': DDPStrategy(find_unused_parameters=False, static_graph=True, gradient_as_bucket_view=True),
        }
    elif cfg.train.strategy == 'cpu':
        trainer_args = {}

    trainer = Trainer(
        limit_val_batches=cfg.train.limit_val_batches,  # to avoid tensorboard issue
        logger=tb_logger,
        **trainer_args,
    )

    trainer.validate(model=task, dataloaders=dm.val_dataloader(), ckpt_path=ckpt_path)


def main():
    # parse arg and cfg
    args = parse_args()
    cfg = load_config(args)

    # set seed
    seed_everything(cfg.seed)

    ckpt_path = cfg.ckpt_path
    if cfg.val.val_only:
        val(args, cfg, ckpt_path)
    else:
        if cfg.train.enable:
            ckpt_path = train(args, cfg)
        if cfg.test.enable:
            test(args, cfg, ckpt_path)


if __name__ == "__main__":
    main()


'''
lta_sf_video:

v1:

python -m scripts.run --cfg configs/ego4d/sf_video.yaml --exp_name ego4d/sf_video \
    train.enable False test.enable True \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/sf_video/checkpoints/epoch=23-step=39048.ckpt

python -m scripts.run --cfg configs/ego4d/sf_video.yaml --exp_name ego4d/sf_video_lr1e-2_epoch5 \
    solver.lr 1e-2 solver.num_epochs 5

    
python -m scripts.run --cfg configs/ego4d/sf_video_reproduce.yaml --exp_name ego4d/sf_video_reproduce
val_as_test:
python -m scripts.run --cfg configs/ego4d/sf_video_reproduce.yaml --exp_name ego4d/sf_video_reproduce \
    train.enable False test.enable True \
    data.test_anno_path data/ego4d/annotations/fho_lta_val.json \
    test.gen_logits True \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/sf_video_reproduce/checkpoints/epoch=23-step=9768.ckpt
val_only:
python -m scripts.run --cfg configs/ego4d/sf_video_reproduce.yaml --exp_name ego4d/sf_video_reproduce \
    val.val_only True \
    num_gpus 2 \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/sf_video_reproduce/checkpoints/epoch=23-step=9768_text.ckpt
python -m scripts.run --cfg configs/ego4d/sf_video_reproduce.yaml --exp_name ego4d/sf_video_reproduce \
    val.val_only True data.examples_to_keep sample_4k.json \
    num_gpus 2 \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/sf_video_reproduce/checkpoints/epoch=23-step=9768_text.ckpt


v2:

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4dv2/sf_video_reproduce.yaml --exp_name ego4dv2/sf_video_reproduce_lr1e-3_epoch20
python -m scripts.run --cfg configs/ego4dv2/sf_video_reproduce.yaml --exp_name ego4dv2/sf_video_reproduce_lr1e-3_epoch20 \
    train.enable False test.enable True \
    data.base_path /gpfs/data/csun45/cfu17/ego4d_fho_data/v1/clips_low_res \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4dv2/sf_video_reproduce_lr1e-3_epoch20/checkpoints/epoch=17-step=19764.ckpt

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4dv2/sf_video_reproduce.yaml --exp_name ego4dv2/sf_video_reproduce_lr1e-3_epoch20_right 

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4dv2/sf_video_reproduce.yaml --exp_name ego4dv2/sfv2_video_reproduce_lr1e-3_epoch20








lta_image:

v1:
python -m scripts.run --cfg configs/ego4d/image.yaml --exp_name ego4d/image_lr5e-3_epoch30
python -m scripts.run --cfg configs/ego4d/image.yaml --exp_name ego4d/image_lr2e-2_epoch30 \
    solver.lr 2e-2
python -m scripts.run --cfg configs/ego4d/image.yaml --exp_name ego4d/image_pos0.5_enc0.5_wd1e-3_layer1 \
    model.pte.pos_dropout 0.5 model.pte.enc_dropout 0.5 model.pte.num_layers 1 solver.weight_decay 1e-3
python -m scripts.run --cfg configs/ego4d/image.yaml --exp_name ego4d/image_pos0.5_enc0.5_wd1e-3_layer2_dropimg0.5 \
    model.pte.pos_dropout 0.5 model.pte.enc_dropout 0.5 model.pte.num_layers 2 model.drop_img 0.5 solver.weight_decay 1e-3
python -m scripts.run --cfg configs/ego4d/image.yaml --exp_name ego4d/image_pos0.5_enc0.5_wd1e-3_layer2_randimg \
    model.pte.pos_dropout 0.5 model.pte.enc_dropout 0.5 model.pte.num_layers 2 solver.weight_decay 1e-3
python -m scripts.run --cfg configs/ego4d/image_reproduce.yaml --exp_name ego4d/reproduce
python -m scripts.run --cfg configs/ego4d/image_reproduce_in8.yaml --exp_name ego4d/image_reproduce_in8
python -m scripts.run --cfg configs/ego4d/image_reproduce.yaml --exp_name ego4d/reproduce seed 2
python -m scripts.run --cfg configs/ego4d/image_reproduce.yaml --exp_name ego4d/reproduce seed 3

v2:
python -m scripts.run --cfg configs/ego4dv2/image_dino.yaml --exp_name ego4dv2/image_dino \
    num_gpus 2
python -m scripts.run --cfg configs/ego4dv2/image_dino.yaml --exp_name ego4dv2/image_dino_lr2e-3 num_gpus 2 \
    solver.lr 2e-3
python -m scripts.run --cfg configs/ego4dv2/image_dino.yaml --exp_name ego4dv2/image_dino_lr5e-4 num_gpus 2 \
    solver.lr 5e-4
python -m scripts.run --cfg configs/ego4dv2/image_clip.yaml --exp_name ego4dv2/image_clip_base2048 num_gpus 2 model.base_feat_size 2048
python -m scripts.run --cfg configs/ego4dv2/image_clip.yaml --exp_name ego4dv2/image_clip_base2048_lr2e-3 num_gpus 2 model.base_feat_size 2048 solver.lr 2e-3
python -m scripts.run --cfg configs/ego4dv2/image_clip.yaml --exp_name ego4dv2/image_clip_base2048_lr5e-3 num_gpus 2 model.base_feat_size 2048 solver.lr 5e-3

test:
python -m scripts.run --cfg configs/ego4dv2/image_dino.yaml --exp_name ego4dv2/image_dino \
    train.enable False test.enable True \
    data.base_path /gpfs/data/csun45/cfu17/ego4d_fho_data_v2/v2/clips_low_res \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4dv2/image_dino/checkpoints/epoch=31-step=36992.ckpt
val_only:
python -m scripts.run --cfg configs/ego4dv2/image_dino.yaml --exp_name ego4dv2/image_dino \
    val.val_only True \
    data.val_anno_path data/ego4d/annotations/fho_lta_val.json \
    num_gpus 2 \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4dv2/image_dino/checkpoints/epoch=31-step=36992.ckpt
    




lta_objects:
python -m scripts.run --cfg configs/ego4dv2/object_dino.yaml --exp_name ego4dv2/object_dino_lr1e-3_pos0.3_base2048 num_gpus 2 model.base_feat_size 2048
python -m scripts.run --cfg configs/ego4dv2/object_dino.yaml --exp_name ego4dv2/object_dino_lr2e-3_pos0.3_base2048 num_gpus 2 solver.lr 2e-3 model.base_feat_size 2048
python -m scripts.run --cfg configs/ego4dv2/object_dino.yaml --exp_name ego4dv2/object_dino_lr5e-4_pos0.3_base2048 num_gpus 2 solver.lr 5e-4 model.base_feat_size 2048




lta_v_i_o:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4dv2/sf_video_image_object_clip.yaml --exp_name ego4dv2/sf_video_image_object_clip_lr1e-3_epoch30 solver.num_epochs 30





lta_text:

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/text.yaml --exp_name ego4d/text_lr1e-3_epoch30_text512_base1024_in8 \
    solver.lr 1e-3 \
    model.text_feat_size 512 model.base_feat_size 1024

val_only:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/text.yaml --exp_name ego4d/text_lr1e-3_epoch30_text512_base1024_in8 \
    model.text_feat_size 512 model.base_feat_size 1024 \
    val.val_only True \
    data.val_anno_path data/ego4d/annotations/fho_lta_val_recog.json \
    num_gpus 4 \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/text_lr1e-3_epoch30_text512_base1024_in8/checkpoints/epoch=20-step=9009.ckpt

val_ony:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/text.yaml --exp_name ego4d/text_lr1e-3_epoch30_text512_base1024 \
    model.text_feat_size 512 model.base_feat_size 1024 \
    val.val_only True \
    data.val_anno_path data/ego4d/annotations/fho_lta_val_recog.json \
    num_gpus 4 \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/text_lr1e-3_epoch30_text512_base1024/checkpoints/epoch=17-step=7326.ckpt

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/text.yaml --exp_name ego4d/text_lr1e-3_epoch30_text512_base1024_layer3_in10 \
    model.text_feat_size 512 model.base_feat_size 1024 model.pte.num_layers 3 \
    val.val_only True \
    data.val_anno_path data/ego4d/annotations/fho_lta_val_recog.json \
    num_gpus 4 \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/text_lr1e-3_epoch30_text512_base1024_layer3_in10/checkpoints/epoch=28-step=9918.ckpt



lta_pred_text:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/image_pred_in8.yaml --exp_name ego4d/image_pred_in8_lr5e-4_droptext0.3 data.prediction_path data/ego4d/fake_all.json

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/image_pred_in8.yaml --exp_name ego4d/image_pred_in8_lr1e-3_droptext0.3 data.prediction_path data/ego4d/fake_all.json \
solver.lr 1e-3

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/image_pred_in8.yaml --exp_name ego4d/image_pred_in8_lr1e-3_droptext1.0 data.prediction_path data/ego4d/fake_all.json \
solver.lr 1e-3 model.drop_text 1.0

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/image_pred_in8.yaml --exp_name ego4d/image_pred_in8_lr1e-3_droptext0.8 data.prediction_path data/ego4d/fake_all.json \
solver.lr 1e-3 model.drop_text 0.8

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/image_pred_in8.yaml --exp_name ego4d/image_pred_in8_lr1e-3_droptext0.6 data.prediction_path data/ego4d/fake_all.json \
solver.lr 1e-3 model.drop_text 0.6

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/image_pred_in8.yaml --exp_name ego4d/image_pred_in8_lr1e-3_droptext0.1_np5 data.prediction_path data/ego4d/fake_all.json \
solver.lr 1e-3 model.drop_text 0.1 data.num_pred_seqs 5

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/image_pred_in8.yaml --exp_name ego4d/image_pred_in8_lr1e-3_droptext0.3_np5 data.prediction_path data/ego4d/fake_all.json \
solver.lr 1e-3 model.drop_text 0.3 data.num_pred_seqs 5

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/image_pred_in8.yaml --exp_name ego4d/image_pred_in8_lr1e-3_droptext0.6_np5 data.prediction_path data/ego4d/fake_all.json \
solver.lr 1e-3 model.drop_text 0.6 data.num_pred_seqs 5

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/image_pred_in8.yaml --exp_name ego4d/image_pred_in8_lr1e-3_droptext0.8_np5 data.prediction_path data/ego4d/fake_all.json \
solver.lr 1e-3 model.drop_text 0.8 data.num_pred_seqs 5

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/image_pred_in8.yaml --exp_name ego4d/image_pred_in8_lr1e-3_droptext0.6_key \
    data.examples_to_keep data/ego4d/fake_all.json data.prediction_path data/ego4d/fake_all.json \
    solver.lr 1e-3 model.drop_text 0.6
    

    
lta_gaze:
NCCL_P2P_DISABLE="1"  python -m scripts.run --cfg configs/gaze_lta/image_clip.yaml --exp_name gaze_lta/image_clip_loc_lr2e-4_e150 solver.lr 2e-4 solver.num_epochs 150

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze_lta/image_clip_optim.yaml --exp_name gaze_lta/image_clip_loc_lr2e-4_e150_optim solver.lr 2e-4 solver.num_epochs 150

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze_lta/image_clip_optim.yaml --exp_name gaze_lta/image_clip_loc_lr1e-4_e150_optim solver.lr 1e-4 solver.num_epochs 150

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze_lta/image_clip_optim.yaml --exp_name gaze_lta/image_clip_loc_lr5e-5_e150_optim solver.lr 5e-5 solver.num_epochs 150

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze_lta/image_clip_optim.yaml --exp_name gaze_lta/image_clip_loc_lr2e-5_e150_optim solver.lr 2e-5 solver.num_epochs 150

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze_lta/image_clip_optim.yaml --exp_name gaze_lta/image_clip_loc_lr1e-4_e150_optim_posdrop0.3 model.pte.pos_dropout 0.3 solver.lr 1e-4 solver.num_epochs 50


NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze_lta/image_clip_optim.yaml --exp_name gaze_lta/image_clip_loc_lr1e-4_e150_optim_posdrop0.1_dropimg0.1 model.pte.pos_dropout 0.1 solver.lr 1e-4 solver.num_epochs 50 model.drop_img 0.1

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze_lta/image_clip_optim.yaml --exp_name gaze_lta/image_clip_loc_lr1e-4_e150_optim_posdrop0.3_dropimg0.3 model.pte.pos_dropout 0.3 solver.lr 1e-4 solver.num_epochs 50 model.drop_img 0.3

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze_lta/image_clip_optim.yaml --exp_name gaze_lta/image_clip_loc_lr1e-4_e150_optim_posdrop0.3_dropimg0.3_metric model.pte.pos_dropout 0.3 solver.lr 1e-4 solver.num_epochs 50 model.drop_img 0.3

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze_lta/image_clip_optim.yaml --exp_name gaze_lta/image_clip_loc_lr1e-4_e150_optim_posdrop0.3_dropimg0.3_zero_metric model.pte.pos_dropout 0.3 solver.lr 1e-4 solver.num_epochs 50 model.drop_img 0.3


NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze_lta/image_clip_optim.yaml --exp_name gaze_lta/image_clip_loc_lr1e-4_e150_optim_posdrop0.5 model.pte.pos_dropout 0.5 solver.lr 1e-4 solver.num_epochs 50


video:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze_lta/video.yaml --exp_name gaze_lta/video_loc_lr1e-4_pos0.3_seg30 model.pte.pos_dropout 0.3 gaze.max_num_segments 30




recognition_sf_video:
python -m scripts.run --cfg configs/ego4d/recognition_sf_video.yaml --exp_name ego4d/recognition_sf_video_lr5e-3_epoch15

python -m scripts.run --cfg configs/ego4d/recognition_sf_video_reproduce.yaml --exp_name ego4d/recognition_sf_video_reproduce_lr1e-2_epoch30 \
    num_gpus 2
python -m scripts.run --cfg configs/ego4d/recognition_sf_video_reproduce.yaml --exp_name ego4d/recognition_sf_video_reproduce_lr1e-2_epoch30 \
    train.enable False test.enable True \
    test.gen_logits True \
    data.test_anno_path data/ego4d/annotations/fho_lta_val.json \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/recognition_sf_video_reproduce_lr1e-2_epoch30/checkpoints/epoch=13-step=9814.ckpt

python -m scripts.run --cfg configs/ego4d/recognition_sf_video_reproduce_ft.yaml --exp_name ego4d/recognition_sf_video_reproduce_ft_lr1e-2_epoch30






recognition_image:
python -m scripts.run --cfg configs/ego4d/recognition_image.yaml --exp_name ego4d/recognition_image_lr5e-3_epoch30

python -m scripts.run --cfg configs/ego4d/recognition_image.yaml --exp_name ego4d/recognition_image_lr1e-2_epoch30 \
    solver.lr 1e-2 

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/recognition_image_clip.yaml --exp_name ego4d/recognition_image_clip_lr1e-2_pos0.3
val_only:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/recognition_image_clip.yaml --exp_name ego4d/recognition_image_clip_lr1e-2_pos0.3
    val.val_only True \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/recognition_image_clip_lr1e-2_pos0.3/checkpoints/epoch=10-step=8118.ckpt
val_as_test:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/recognition_image_clip.yaml --exp_name ego4d/recognition_image_clip_lr1e-2_pos0.3 \
    train.enable False test.enable True \
    data.test_anno_path data/ego4d/annotations/fho_lta_val.json \
    test.gen_logits True \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/recognition_image_clip_lr1e-2_pos0.3/checkpoints/epoch=10-step=8118.ckpt
train_as_test:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/recognition_image_clip.yaml --exp_name ego4d/recognition_image_clip_lr1e-2_pos0.3 \
    train.enable False test.enable True \
    data.test_anno_path data/ego4d/annotations/fho_lta_train.json \
    test.gen_logits True \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/recognition_image_clip_lr1e-2_pos0.3/checkpoints/epoch=10-step=8118.ckpt
test_as_test:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4d/recognition_image_clip.yaml --exp_name ego4d/recognition_image_clip_lr1e-2_pos0.3 \
    train.enable False test.enable True \
    data.test_anno_path data/ego4d/annotations/fho_lta_test_unannotated.json \
    test.gen_logits True \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4d/recognition_image_clip_lr1e-2_pos0.3/checkpoints/epoch=10-step=8118.ckpt
    
v2:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4dv2/recognition_image_clip.yaml --exp_name ego4dv2/recognition_image_clip_lr1e-2_pos0.3
val_only:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4dv2/recognition_image_clip.yaml --exp_name ego4dv2/recognition_image_clip_lr1e-2_pos0.3 \
    val.val_only True \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4dv2/recognition_image_clip_lr1e-2_pos0.3/checkpoints/epoch=7-step=15984.ckpt
val_as_test:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4dv2/recognition_image_clip.yaml --exp_name ego4dv2/recognition_image_clip_lr1e-2_pos0.3 \
    train.enable False test.enable True \
    data.test_anno_path /gpfs/data/csun45/cfu17/ego4d_fho_data_v2/v2/annotations/fho_lta_val.json \
    test.gen_logits True \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4dv2/recognition_image_clip_lr1e-2_pos0.3/checkpoints/epoch=7-step=15984.ckpt
train_as_test:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4dv2/recognition_image_clip.yaml --exp_name ego4dv2/recognition_image_clip_lr1e-2_pos0.3 \
    train.enable False test.enable True \
    data.test_anno_path /gpfs/data/csun45/czhan164/ego4dv2_files/fho_lta_train.json \
    test.gen_logits True \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4dv2/recognition_image_clip_lr1e-2_pos0.3/checkpoints/epoch=7-step=15984.ckpt
test_as_test:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/ego4dv2/recognition_image_clip.yaml --exp_name ego4dv2/recognition_image_clip_lr1e-2_pos0.3 \
    train.enable False test.enable True \
    data.test_anno_path /gpfs/data/csun45/cfu17/ego4d_fho_data_v2/v2/annotations/fho_lta_test_unannotated.json \
    test.gen_logits True \
    ckpt_path /users/czhan164/anticipation/lightning_logs/ego4dv2/recognition_image_clip_lr1e-2_pos0.3/checkpoints/epoch=7-step=15984.ckpt


# gaze
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip.yaml --exp_name gaze/recognition_image_clip_lr1e-2_epoch30_pos0.3

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip.yaml --exp_name gaze/recognition_image_clip_lr1e-2_epoch50_pos0.3 solver.num_epochs 50

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip.yaml --exp_name gaze/recognition_image_clip_lr1e-2_epoch50_pos0.5 solver.num_epochs 50 model.pte.pos_dropout 0.5

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip.yaml --exp_name gaze/recognition_image_clip_lr1e-2_epoch50_pos0.1 solver.num_epochs 50 model.pte.pos_dropout 0.1

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip.yaml --exp_name gaze/recognition_image_clip_lr1e-2_epoch50_pos0.1_layer2 solver.num_epochs 50 model.pte.pos_dropout 0.1 model.pte.num_layers 2

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip.yaml --exp_name gaze/recognition_image_clip_lr1e-2_epoch50_pos0.1_layer1 solver.num_epochs 50 model.pte.pos_dropout 0.1 model.pte.num_layers 1

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip.yaml --exp_name gaze/recognition_image_clip_lr1e-2_epoch50_pos0.3_image8 solver.num_epochs 50 data.image.num_images_per_segments 8

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip.yaml --exp_name gaze/recognition_image_clip_lr1e-2_epoch50_pos0.3_image2 solver.num_epochs 50 data.image.num_images_per_segments 2

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip_avt.yaml --exp_name gaze/recognition_image_clip_avt_lr1e-2_epoch50_pos0.3_dropimg0.3 solver.num_epochs 50

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip_avt.yaml --exp_name gaze/recognition_image_clip_avt_lr1e-2_epoch50_pos0.3_dropimg0.3_numimg20 solver.num_epochs 50 data.image.num_images_per_segment 20

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip_avt.yaml --exp_name gaze/recognition_image_clip_avt_lr1e-2_epoch50_pos0.3_dropimg0.5_numimg20 solver.num_epochs 50 data.image.num_images_per_segment 20 model.drop_img 0.5

NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip_avt.yaml --exp_name gaze/recognition_image_clip_avt_lr1e-2_epoch50_pos0.3_dropimg0.1_numimg20 solver.num_epochs 50 data.image.num_images_per_segment 20 model.drop_img 0.1


val_as_test:
NCCL_P2P_DISABLE="1" python -m scripts.run --cfg configs/gaze/recognition_image_clip_avt.yaml --exp_name gaze/recognition_image_clip_avt_lr1e-2_epoch50_pos0.3_dropimg0.3_numimg20 solver.num_epochs 50 data.image.num_images_per_segment 20 \
    train.enable False test.enable True \
    test.gen_logits True \
    ckpt_path /users/czhan164/anticipation/lightning_logs/gaze/recognition_image_clip_avt_lr1e-2_epoch50_pos0.3_dropimg0.3_numimg20/checkpoints/epoch=44-step=11655.ckpt
'''




'''
debug:
python -m scripts.run --cfg configs/ego4d/sf_video.yaml --exp_name ego4d/debud_acc \
    num_gpus 1 train.limit_train_batches 0.005 train.limit_val_batches 0.01 \
    train.batch_size 4 val.batch_size 4
'''


'''
check submit file:
python -m src.utils.eval_util
'''



'''
python -m scripts.run --cfg configs/ego4d/text.yaml --exp_name ego4d/text_lr1e-3_epoch30_text512_base1024_layer2 \
    num_gpus 2 solver.lr 1e-3 \
    model.text_feat_size 512 model.base_feat_size 1024 model.pte.num_layers 2

python -m scripts.run --cfg configs/ego4d/text.yaml --exp_name ego4d/text_lr1e-3_epoch30_text512_base1024_layer3 \
    num_gpus 2 solver.lr 1e-3 \
    model.text_feat_size 512 model.base_feat_size 1024 model.pte.num_layers 3

python -m scripts.run --cfg configs/ego4d/text_input10.yaml --exp_name ego4d/text_lr1e-3_epoch30_text512_base1024_layer2_in10 \
    num_gpus 2 solver.lr 1e-3 \
    model.text_feat_size 512 model.base_feat_size 1024 model.pte.num_layers 2

python -m scripts.run --cfg configs/ego4d/text_input10.yaml --exp_name ego4d/text_lr1e-3_epoch30_text512_base1024_layer3_in10 \
    num_gpus 2 solver.lr 1e-3 \
    model.text_feat_size 512 model.base_feat_size 1024 model.pte.num_layers 3


python -m scripts.run --cfg configs/ego4d/text_input8.yaml --exp_name ego4d/text_lr1e-3_epoch30_text512_base1024_layer3_in10 \
    solver.lr 1e-3 model.text_feat_size 512 model.base_feat_size 1024 model.pte.num_layers 3 seed 2
python -m scripts.run --cfg configs/ego4d/text_input8.yaml --exp_name ego4d/text_lr1e-3_epoch30_text512_base1024_layer3_in10 \
    solver.lr 1e-3 model.text_feat_size 512 model.base_feat_size 1024 model.pte.num_layers 3 seed 3



python -m scripts.run --cfg configs/ego4d/text_input10.yaml --exp_name ego4d/text_lr1e-3_epoch30_text512_base1024_in10 \
    num_gpus 2 solver.lr 1e-3 \
    model.text_feat_size 512 model.base_feat_size 1024

python -m scripts.run --cfg configs/ego4d/text.yaml --exp_name ego4d/text_lr1e-3_epoch30_text1024_base1024 \
    num_gpus 2 solver.lr 1e-3 \
    model.text_feat_size 1024 model.base_feat_size 1024

python -m scripts.run --cfg configs/gaze_lta/video_clip.yaml --exp_name gaze_lta/vc_1 num_gpus 2 solver.lr 2e-2 solver.num_epochs 50
'''