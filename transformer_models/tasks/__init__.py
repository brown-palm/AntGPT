from .video_classification_task import VideoClassificationTask
from .video_multiclasscls_task import VideoMultiClassClsTask


def load_task(cfg, steps_in_epoch=1):
    if cfg.multicls.enable:
        return VideoMultiClassClsTask(cfg, steps_in_epoch)
    else:
        return VideoClassificationTask(cfg, steps_in_epoch)
