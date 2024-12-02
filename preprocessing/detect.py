import os
import numpy as np
from suite2p.run_s2p import pipeline
from suite2p import registration
from suite2p.io import BinaryFile


def detect_rois(ops, fs=17):
    detection_ops = {
        "save_path": ops['save_path'],
        "tau": 0.7,
        "fs": fs,
        "preclassify": 0.0,
        "batch_size": 500,
        "aspect": 1,

        "roidetect": True,
        "spikedetect": True,
        "sparse_mode": True,
        "spatial_scale": 0,
        "connected": True,
        "nbinned": 5000,
        "max_iterations": 20,
        "threshold_scaling": 1.0,
        "max_overlap": 0.75,
        "high_pass": 100,
        "spatial_hp_detect": 25,
        "denoise": False,

        "anatomical_only": 2,
        "diameter": 0,
        "cellprob_threshold": 0.0,
        "flow_threshold": 1.5,
        "spatial_hp_cp": 0,
        "pretrained_model": "cyto",

        "soma_crop": True,
        "neuropil_extract": True,
        "inner_neuropil_radius": 2,
        "min_neuropil_pixels": 350,
        "lam_percentile": 50.,
        "allow_overlap": False,
        "use_builtin_classifier": False,
        "classifier_path": "",

        "chan2_thres": 0.65,  # minimum for detection of brightness on channel 2

        "baseline": "maximin",
        "win_baseline": 60.,
        "sig_baseline": 10.,
        "prctile_baseline": 8.,
        "neucoeff": 0.7
    }

    ops = {**ops, **detection_ops}
    ops["meanImgE"] = registration.compute_enhanced_mean_image(ops["meanImg"].astype(np.float32), ops)

    reg_file, l_x, l_y = ops['reg_file'], ops['Lx'], ops['Ly']
    with BinaryFile(Ly=l_y, Lx=l_x, filename=reg_file) as f_reg:
        ops_ = pipeline(f_reg=f_reg, f_raw=None, f_reg_chan2=None, f_raw_chan2=None,
                        run_registration=False, ops=detection_ops, stat=None)

    return {**ops, **detection_ops, **ops_}
