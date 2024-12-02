import os
from glob import glob
import suite2p
from suite2p import io


def tif_to_bin(root_path):
    if os.path.exists(os.path.join(root_path, 'suite2p')) and \
            os.path.isfile(os.path.join(root_path, 'suite2p', 'plane0', 'data_raw.bin')):
        print('raw bin files already exist, skipping...')
        return
    else:
        print('raw bin files not found, starting to divide channels...')

    ops = {
        "nplanes": 1,
        "nchannels": 2,
        "keep_movie_raw": True,
        "look_one_level_down": False,
        "batch_size": 500,
        "functional_chan": 2,
        "do_registration": True,
        "nonrigid": True,
        "move_bin": True,
        "fs": 17,
        "force_sktiff": False
    }

    db = {
        'save_path0': root_path,
        'save_folder': "suite2p",
        'data_path': [os.path.join(root_path, 'raw')],
        'input_format': 'tif',
    }

    io.tiff_to_binary({**ops, **db})
