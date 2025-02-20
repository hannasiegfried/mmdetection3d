import torch
import yaml
from fw_lidar.dsp.single_stage.single_stage_model import SingleStageModel

def load_cfg(cfg_yaml_path):
    with open(cfg_yaml_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

def load_model(path):
    model_cfg_path = f"{path}/configs/single_stage_swinunet_centerfov.yaml"
    weights_path = f"{path}/logs/2025-01-21_16-11-13_carla_singlestage_dist_single_stage_swinunet_centerfov_notag/state_dict_ep070.pth"

    model_cfg = load_cfg(model_cfg_path)
    loss_cfg = model_cfg["Network"]["loss"]
    model = SingleStageModel(model_cfg["Network"], loss_cfg)
    weights = torch.load(weights_path)
    weights_stripped = {}
    for k, v in weights.items():
        k_stripped = ".".join(k.split(".")[1:])
        weights_stripped[k_stripped] = v
    model.load_state_dict(weights_stripped)

    return model