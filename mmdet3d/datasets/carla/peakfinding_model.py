import torch
import yaml
import numpy as np
from fw_lidar.dsp.single_stage.single_stage_model import SingleStageModel

def load_cfg(cfg_yaml_path):
    with open(cfg_yaml_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

def load_model(path):
    model_cfg_path = f"{path}/configs/single_stage_swinunet_centerfov.yaml"
    weights_path = None
    #weights_path = f"{path}/logs/2025-04-16_17-51-23_carla_singlestage_single_stage_swinunet_centerfov_tag/state_dict_ep018.pth"
    view_dir = np.load("/home/hasiegf/thesis/sensor_specs/view_direction_carla_60deg.npy")
    
    cfg = load_cfg(model_cfg_path)
    model_cfg = cfg["Network"]
    view_dir = torch.from_numpy(view_dir).float()
    model_cfg["params"].update({"view_encoding": view_dir})
    model = SingleStageModel(model_cfg, model_cfg["loss"])

    if weights_path:
        weights = torch.load(weights_path)
        weights_stripped = {}
        for k, v in weights.items():
            k_stripped = ".".join(k.split(".")[1:])
            weights_stripped[k_stripped] = v
        model.load_state_dict(weights, strict=False)
        
    model.cuda()
    return model