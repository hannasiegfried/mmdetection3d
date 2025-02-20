import numpy as np
from fw_lidar.peak_finding import PFScipy
from fw_lidar.dsp.load_cfg import load_cfg  

class PeakFinding:
    def __init__(self):
        self.pf_cfg = load_cfg("mmdet3d/datasets/carla/baseline_peakfinding_carla.yaml")
        self.pf = PFScipy(**self.pf_cfg["params"])

    def forward(self, waveform):
        tofs, tots = [], []
        tof, tot = self.pf.get_tof_tot_values(waveform)
        tofs.append(tof)
        tots.append(tot)
        return dict(tof=np.asarray(tofs), tot=np.asarray(tots))