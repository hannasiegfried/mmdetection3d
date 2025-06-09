import torch
import yaml
import numpy as np
from fw_lidar.dsp.single_stage.single_stage_model import SingleStageModel

def load_cfg(cfg_yaml_path):
    with open(cfg_yaml_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

def load_weights(model):
    #weights_path = None
    #weights_path = f"{path}/logs/2025-04-30_22-25-26_carla_singlestage_single_stage_swinunet_centerfov_tag/state_dict_ep014.pth"
    weights_path = '/media/hasiegf/data/mmdet3d/out/carla/transformer/queries/1/relu/epoch_80.pth'
    if weights_path:
        weights = torch.load(weights_path, weights_only=False)
        weights_stripped = {}
        weights = weights["state_dict"]
        for k, v in weights.items():
            k_stripped = ".".join(k.split(".")[1:])
            weights_stripped[k_stripped] = v
        model.load_state_dict(weights_stripped, strict=False)
    return model

def load_model(path):
    model_cfg_path = f"{path}/configs/single_stage_swinunet_centerfov.yaml"
    view_dir = np.load("/home/hasiegf/thesis/sensor_specs/view_direction_carla_60deg.npy")
    
    cfg = load_cfg(model_cfg_path)
    model_cfg = cfg["Network"]
    view_dir = torch.from_numpy(view_dir).float()
    model_cfg["params"].update({"view_encoding": view_dir})
    model = SingleStageModel(model_cfg, model_cfg["loss"])
    model = load_weights(model)
    model.cuda()
    return model