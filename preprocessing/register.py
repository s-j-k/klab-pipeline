import os
import numpy as np
from suite2p.io import BinaryFile
from suite2p import registration


def register_from_bin(session_path, batch_size=2000):
    plane_path = os.path.join(session_path, 'suite2p', 'plane0')
    ops_path = os.path.join(plane_path, 'ops.npy')
    ops = np.load(ops_path, allow_pickle=-True).item()
    l_x, l_y = ops['Lx'], ops['Ly']

    registration_ops = {
        "data_path": [plane_path],
        "save_path": plane_path,
        "functional_chan": 1,
        "frames_include": -1,
        "do_bidiphase": False,
        "bidiphase": 0,
        "bidi_corrected": False,

        "do_registration": True,
        "align_by_chan": 1,
        "nimg_init": 300,
        "batch_size": batch_size,
        "maxregshift": 0.1,
        "smooth_sigma": 1.15,
        "smooth_sigma_time": 0.0,
        "keep_movie_raw": True,
        "two_step_registration": False,
        "reg_tif": False,
        "reg_tif_chan2": False,
        "subpixel": 10,
        "th_badframes": 1.0,
        "norm_frames": True,
        "force_refImg": False,
        "pad_fft": False,

        "1Preg": False,

        "nonrigid": True,
        "block_size": [128, 128],
        "snr_thresh": 1.2,
        "maxregshiftNR": 5.0
    }

    raw_file_chan1 = os.path.join(plane_path, f'data_raw.bin')
    raw_file_chan2 = os.path.join(plane_path, f'data_chan2_raw.bin')
    reg_file_chan1 = os.path.join(plane_path, f'data.bin')
    reg_file_chan2 = os.path.join(plane_path, f'data_chan2.bin')

    with BinaryFile(Ly=l_y, Lx=l_x, filename=raw_file_chan1) as f_raw_1:
        n_frames = f_raw_1.shape[0]
        with BinaryFile(Ly=l_y, Lx=l_x, filename=raw_file_chan2) as f_raw_2, \
                BinaryFile(Ly=l_y, Lx=l_x, filename=reg_file_chan1, n_frames=n_frames) as f_reg_1, \
                BinaryFile(Ly=l_y, Lx=l_x, filename=reg_file_chan2, n_frames=n_frames) as f_reg_2:
            registration_outputs = registration.registration_wrapper(
                f_reg=f_reg_1, f_raw=f_raw_1, f_reg_chan2=f_reg_2, f_raw_chan2=f_raw_2,
                align_by_chan2=False, ops=registration_ops)

    ops = registration.save_registration_outputs_to_ops(registration_outputs, ops)
