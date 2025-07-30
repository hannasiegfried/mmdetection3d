import torch
import yaml
import numpy as np
from fw_lidar.dsp.single_stage.single_stage_model import SingleStageModel

def load_cfg(cfg_yaml_path):
    with open(cfg_yaml_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

def load_weights(model):
    weights_path = None
    # normal weights
    # Decoder approach pretrained
    #weights_path = "/home/hasiegf/thesis/fw_lidar/logs/2025-07-17_10-11-02_carla_singlestage_dist_single_stage_swinunet_centerfov_tag/state_dict_ep079.pth"
    # Threshold approach pretrained
    #weights_path = f"/home/hasiegf/thesis/fw_lidar/logs/2025-07-15_12-22-31_carla_singlestage_dist_single_stage_swinunet_centerfov_tag/state_dict_ep079.pth"
    #weights_path = '/media/hasiegf/data/mmdet3d/out/carla/transformer/queries/1/relu/epoch_20.pth'
    #weights_path = '/media/hasiegf/data/mmdet3d/out/carla/transformer/queries/1/relu/input/pred_score/epoch_80.pth'
    if weights_path:
        weights = torch.load(weights_path, weights_only=False)
        weights_stripped = {}
        #weights = weights["state_dict"]
        for k, v in weights.items():
            k_stripped = ".".join(k.split(".")[1:])
            weights_stripped[k_stripped] = v
        #weights_stripped
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