import yaml
import shutil
import suite2p
from glob import glob
from pathlib import Path
from sbxreader import sbx_memmap
from PIL import Image
import tifffile
import os
import numpy as np
from suite2p import io, extraction, classification
from matplotlib import pyplot as plt

import preprocessing

#this code is for two channels

def fn_get_cellprofiler_masks(path):
    _m = []
    for mask_file_name in glob(os.path.join(path, '*.tiff')):
        mask = np.array(Image.open(mask_file_name))
        ypix, xpix = np.nonzero(mask)
        lam = np.ones_like(ypix) / len(ypix)

        xmed, ymed = np.median(xpix), np.median(ypix)
        radius = len(ypix) ** 0.5
        radius /= (np.pi ** 0.5) / 2

        _m.append({'xpix': xpix, 'ypix': ypix, 'lam': lam, 'radius': radius,
                   'med': [ymed, xmed], 'npix': len(xpix),
                   'aspect_ratio': 1, "compact": 1, "footprint": 1})
    return _m


with open('meta3.yaml', 'r') as file: # this is the metadata file in the local code folder
    all_sessions = list(yaml.safe_load_all(file))

for session in all_sessions:
    if session['done_ch1']:
        continue

    session_path = Path(session['session_path'])

    if session_path.joinpath("suite2p").is_dir():
        try:
            shutil.rmtree(str(session_path.joinpath("suite2p")))
        except PermissionError:
            session_path.joinpath('permerror.log').touch()

    ops = preprocessing.default_ops()
    data_bin_file = session_path.joinpath("suite2p", "plane0", "data.bin")

    if data_bin_file.is_file():
        print(f"session {session['session_path']} is already preprocessed, skipping:")

    fs = session['f_sample']
    if fs == -1:
        if session['input_format'] == "sbx":
            sbx_file = glob(str(session_path.joinpath('raw').joinpath('*.sbx')))
            if len(sbx_file) != 1:
                raise IOError("Directory has multiple sbx files")
            mmap = sbx_memmap(sbx_file[0])
            fs = mmap.metadata['frame_rate'] / 2
            mmap._mmap.close()
        elif session['input_format'] == "tif":
            tif_file=glob(str(session_path.joinpath('raw').joinpath('*.tif')))
            if len(tif_file) != 1:
                raise IOError("Directory has multiple tiff files")
            mmap = tif_memmap(tif_file[0])
            fs = mmap.metadata['frame_rate'] # this is for one plane. if two, div by 2
            mmap._mmap.close()
        else:
            raise NotImplementedError("Extracting sampling rate from metadata is not implemented for this format")

    db = {
        'fast_disk': [],
        'save_path0': str(session_path),
        'data_path': [str(session_path.joinpath('raw'))],
        'input_format': session['input_format'],
        'fs': fs,
        'aspect': session['ppm_x'] / session['ppm_y'],
        'keep_movie_raw': False,
        'batch_size': 1000 
    }

    ops = suite2p.run_s2p(ops, db)

    session['done_ch1'] = True
    with open('meta.yaml', 'w') as file:
        yaml.dump_all(all_sessions, file, default_flow_style=False)

    mean_image = {'chan1': ops['meanImg'], 'chan2': ops['meanImg_chan2']}
    for chan in mean_image:
        im = mean_image[chan]
        im = im.clip(*np.percentile(im, (1, 99)))
        plt.imsave(str(session_path.joinpath(f"mean_image_{chan}-enhanced.tiff")), im, cmap='gray')

# Extracting fluorescence traces from green channel
for session in all_sessions:
    if 'done_ch2' in session.keys() and session['done_ch2']:
        continue

    output_folder = Path(os.path.join(session["session_path"], "suite2p", "plane0", "channel2"))
    os.makedirs(str(output_folder), exist_ok=True)

    ops_file_name = os.path.join(session["session_path"], "suite2p", "plane0", "ops.npy")
    ops = np.load(ops_file_name, allow_pickle=True).item()
    mean_image_buffer = ops['meanImg']
    ops['meanImg'] = ops['meanImg_chan2']
    ops['meanImg_chan2'] = mean_image_buffer

    ops['reg_file'] = str(output_folder.parent.parent.joinpath('data_chan2.bin'))
    ops['reg_file_chan2'] = str(output_folder.parent.parent.joinpath('data.bin'))

    mask_path = os.path.join(session["session_path"], "rois", "mean_image_chan2-enhanced")
    if not os.path.isdir(mask_path):
        session['done_ch2'] = False
        with open('meta.yaml', 'w') as file:
            yaml.dump_all(all_sessions, file, default_flow_style=False)
        continue

    stat = fn_get_cellprofiler_masks(mask_path)
    bin_path = os.path.join(session["session_path"], "suite2p", "plane0")
    with io.BinaryFile(filename=os.path.join(bin_path, "data_chan2.bin"), Lx=ops['Lx'], Ly=ops['Ly']) as f_reg, \
            io.BinaryFile(filename=os.path.join(bin_path, "data.bin"), Lx=ops['Lx'], Ly=ops['Ly']) as f_reg_chan2:
        stat, f, f_neu, f_chan2, f_neu_chan2 = extraction.extraction_wrapper(stat, f_reg, f_reg_chan2=f_reg_chan2,
                                                                             ops={**ops, **{"allow_overlap": True}})

    spks = np.zeros_like(f)
    iscell = classification.classify(stat=stat, classfile=classification.user_classfile)
    redcell = iscell.copy()

    np.save(output_folder.joinpath("ops.npy"), ops)
    np.save(output_folder.joinpath("stat.npy"), stat)
    np.save(output_folder.joinpath("F.npy"), f)
    np.save(output_folder.joinpath("Fneu.npy"), f_neu)
    np.save(output_folder.joinpath("F_chan2.npy"), f_chan2)
    np.save(output_folder.joinpath("Fneu_chan2.npy"), f_neu_chan2)
    np.save(output_folder.joinpath("spks.npy"), spks)
    np.save(output_folder.joinpath("iscell.npy"), iscell)
    np.save(output_folder.joinpath("redcell.npy"), redcell)

    session['done_ch2'] = True
    with open('meta.yaml', 'w') as file:
        yaml.dump_all(all_sessions, file, default_flow_style=False)
