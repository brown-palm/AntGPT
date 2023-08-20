from .model import ClassificationModule

def build_model(cfg):
    return ClassificationModule(cfg)