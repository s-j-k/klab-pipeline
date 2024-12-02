import os
from glob import glob
from suite2p import io
from sbxreader import sbx_memmap


def sbx_to_bin(root_path):
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
        "move_bin": True
    }

    interface = sbx_memmap(glob(os.path.join(root_path, 'raw', '*.sbx'))[0])
    sample_rate = interface.metadata['frame_rate'] / interface.metadata['num_planes']

    db = {
        'save_path0': root_path,
        'save_folder': "suite2p",
        'data_path': [os.path.join(root_path, 'raw')],
        'input_format': 'sbx',
        'fs': sample_rate
    }

    io.sbx_to_binary({**ops, **db}, ndeadcols=-1, ndeadrows=0)
