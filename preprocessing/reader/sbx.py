import os
from glob import glob
import suite2p
from sbxreader import sbx_memmap

ops = suite2p.default_ops()
# ---------- Main setting ----------
ops['nplanes'] = 1
ops['nchannels'] = 2
ops['functional_chan'] = 2
ops['tau'] = 0.7
# 0.7 for GCaMP6f,
# 1.0 for GCaMP6m,
# 1.25-1.5 for GCaMP6s,
# 0.7 for jrgeco https://github.com/MouseLand/suite2p/issues/233

# ---------- file io ----------
ops['delete_bin'] = False
ops['save_folder'] = 'suite2p'
ops['move_bin'] = False

# ---------- output setting ----------
ops['save_nwb'] = False
ops['save_mat'] = False
ops['aspect'] = 2.55/2.66

# ---------- registration ----------
ops['align_by_chan'] = 2
ops['batch_size'] = 500
ops['maxregshift'] = 0.1
ops['keep_movie_raw'] = False
ops['reg_tif'] = False
ops['reg_tif_chan2'] = False

# ---------- cellpose ----------
ops['anatomical_only'] = 2

# ---------- channel 2 ----------
ops['chan2_thres'] = 0.65

data_path = "E:/KishoreLab/Moe/Data/Su/083/session0/raw"
interface = sbx_memmap(glob(os.path.join(data_path, '*.sbx'))[0])
save_path = "E:/KishoreLab/Moe/Data/Su/083/session0"

db = {
    'save_path0': save_path,
    'data_path': [data_path],
    'input_format': 'sbx',
    'fs': interface.metadata['frame_rate'] / interface.metadata['num_planes']
}

suite2p.run_s2p(ops, db)
