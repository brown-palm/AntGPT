import argparse
from .defaults import get_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        help="path to the config file",
        default="not_provided",
        type=str,
    )
    parser.add_argument("--fast_dev_run", action='store_true')
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument(
        "opts",
        help="See configs/defaults.py for all options.",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def load_config(args):
    cfg = get_cfg()
    if args.cfg is not None:
        cfg.merge_from_file(args.cfg)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.exp_name = args.exp_name

    assert cfg.data.output_segments[0] < cfg.data.output_segments[1], 'data.output_segments must be in an increasing order'
    assert cfg.data.image.input_segments[0] < cfg.data.image.input_segments[1], 'data.image.input_segments must be in an increasing order'

    # always set data.input_segments
    if cfg.model.img_feat_size > 0:
        if cfg.data.image.input_from_annotated_segments:
            assert str(cfg.data.image.input_segments) == str(cfg.data.input_segments)
    
    # model sizes consistent with data config
    #assert cfg.model.num_actions_to_predict == cfg.data.output_segments[1] - cfg.data.output_segments[0]
    #assert len(cfg.model.loss_wts_temporal) == cfg.data.output_segments[1] - cfg.data.output_segments[0]

    assert cfg.train.enable ^ cfg.test.enable, 'must choose exactly one from either train.enable or test.enable'
    
    cfg.freeze()
    return cfg
