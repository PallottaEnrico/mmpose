from mmengine.config import Config, DictAction
from mmengine.runner import Runner

cfg = Config.fromfile(
    './src/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
)

cfg.data_root = 'data/gates/'

runner = Runner.from_cfg(cfg)

runner.train()