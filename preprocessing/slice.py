import os
from glob import glob
from .divider import sbx_to_bin, tif_to_bin
import numpy as np
from suite2p.io import BinaryFile
from suite2p.registration import registration_wrapper
from pathlib import Path


class WrongDirectoryStructure(Exception):
    pass


def open_quadrant_binary_files(subject_path, original_folder_name):
    fid1, fid2, dirs = [], [], []
    for q in range(4):
        fov_dir = os.path.join(subject_path, f'{original_folder_name}.{q+1}', 'session0', 'suite2p')
        os.makedirs(fov_dir, exist_ok=True)

        dirs.append(fov_dir)
        fid1.append(open(os.path.join(fov_dir, 'data_chan1_raw.bin'), 'wb'))
        fid2.append(open(os.path.join(fov_dir, 'data_chan2_raw.bin'), 'wb'))
    return fid1, fid2, dirs


def split_channels(path):
    tif_files = glob(os.path.join(path, 'raw', '*.tif'))
    sbx_files = glob(os.path.join(path, 'raw', '*.sbx'))

    if (len(tif_files) == 1) and (len(sbx_files) == 0):
        # if os.path.isdir(os.path.join())
        tif_to_bin(path)
    elif (len(tif_files) == 0) and (len(sbx_files) == 1):
        sbx_to_bin(path)
    else:
        raise WrongDirectoryStructure()


def split_to_quadrants(session_path, batch_size=500):
    subject_path = Path(session_path).parents[1]
    ops_path = os.path.join(session_path, 'suite2p', 'plane0', 'ops.npy')
    ops = np.load(ops_path, allow_pickle=-True).item()
    l_x, l_y = ops['Lx'], ops['Ly']
    c_x, c_y = l_x // 2, l_y // 2

    ops_out = [{} for _ in range(4)]
    ops_out[0]['Lx'], ops_out[0]['Ly'] = c_x, c_y
    ops_out[1]['Lx'], ops_out[1]['Ly'] = c_x, l_y - c_y
    ops_out[2]['Lx'], ops_out[2]['Ly'] = l_x - c_x, c_y
    ops_out[3]['Lx'], ops_out[3]['Ly'] = l_x - c_x, l_y - c_y

    fq1, fq2, fov_dirs = open_quadrant_binary_files(subject_path, Path(session_path).parents[0])
    if os.path.isfile(os.path.join(fov_dirs[0], 'ops.npy')):
        print(f"{fov_dirs[0]} exists. Skipping {session_path} session")
        return

    for fq, chan_str in zip([fq1, fq2], ["", "_chan2"]):
        raw_path = os.path.join(session_path, 'suite2p', 'plane0', f'data{chan_str}_raw.bin')

        with BinaryFile(Ly=l_y, Lx=l_x, filename=raw_path) as f:
            nframes = f.shape[0]
            iblocks = np.arange(0, nframes, batch_size)
            if iblocks[-1] < nframes:
                iblocks = np.append(iblocks, nframes)

            for onset, offset in zip(iblocks[:-1], iblocks[1:]):
                im = np.array(f[onset:offset, :, :])
                fq[0].write(bytearray(im[:, :c_y, :c_x].astype("int16")))
                fq[1].write(bytearray(im[:, :c_y, c_x:].astype("int16")))
                fq[2].write(bytearray(im[:, c_y:, :c_x].astype("int16")))
                fq[3].write(bytearray(im[:, c_y:, c_x:].astype("int16")))

    for fid in [*fq1, *fq2]:
        fid.close()

    for fov_dir, ops_no in zip(fov_dirs, ops_out):
        np.save(os.path.join(fov_dir, 'ops.npy'), ops_no)


