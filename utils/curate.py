import os
import numpy as np
from glob import glob
from PIL import Image
from scipy.ndimage import gaussian_filter1d as smooth
from scipy.stats import ttest_rel as ttest
import statsmodels.stats.multitest as smm

from lites2p import create_masks, BinaryFile, extract_traces
from lites2p.dcnv import preprocess


def during(t, a, b):
    return np.argwhere((t <= b) & (t > a)).flatten()


def fn_get_s2p_cell_masks(path):
    options = np.load(os.path.join(path, 'plane0', 'ops.npy'), allow_pickle=True).item()
    stat = np.load(os.path.join(path, 'plane0', 'stat.npy'), allow_pickle=True)
    masks, _ = create_masks(stat, options['Ly'], options['Lx'], options)
    return masks


def fn_get_cellprofiler_masks(path):
    masks = []
    for mask_file_name in glob(os.path.join(path, '*.tiff')):
        mask = np.array(Image.open(mask_file_name))
        ind = np.ravel_multi_index(np.nonzero(mask), dims=mask.shape)
        lam = mask.flatten()[ind].astype(np.double)
        lam /= lam.sum()

        masks.append((ind, lam))
    return masks


def fn_select_neurons(time, dff, evt,
                      baseline_time_range=(-700, 0),
                      activity_time_range=(0, 700),
                      alpha=0.05):
    selective_neurons = []
    for i_roi in range(dff.shape[2]):
        baseline = dff[:, during(time, *baseline_time_range), i_roi].mean(1)
        activity = dff[:, during(time, *activity_time_range), i_roi].mean(1)

        tone_p_values = []
        for tone in np.unique(evt):
            _, tone_p = ttest(activity[evt == tone], baseline[evt == tone])
            tone_p_values.append(tone_p)
        rejected, p_values_corrected = smm.fdrcorrection(tone_p_values, alpha=alpha)

        if any(rejected):
            selective_neurons.append(i_roi)

    return selective_neurons


def load_data(bin_path, masks, frame_w, frame_h, sigma=2):
    bin1 = BinaryFile(filename=bin_path, Lx=frame_w, Ly=frame_h)
    f, _ = extract_traces(bin1, masks, None, batch_size=500)
    return f


def mk_epochs(mod, onset, t_bound=(-20, 79), bl_bound=(-700, 0), bl_normalize=False):
    fluorescence_trace = mod.data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries'].data[:]
    plane_segmentation = mod.data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']
    sampling_rate = plane_segmentation.imaging_plane.imaging_rate

    fluorescence_trace = preprocess(F=fluorescence_trace,
                                    baseline='minimax',
                                    win_baseline=60,
                                    sig_baseline=3,
                                    fs=sampling_rate,
                                    prctile_baseline=8)

    # create epochs
    frame_range = np.arange(*t_bound)
    trange = frame_range / sampling_rate * 1000
    in_epoch_indices = (onset[:, np.newaxis] + frame_range).astype('int')
    epochs_ = fluorescence_trace[in_epoch_indices, :]

    # baseline normalize the epochs
    if bl_normalize:
        baseline = epochs_[:, during(trange, *bl_bound), :].mean(axis=1, keepdims=True)
        epochs_ = epochs_ - baseline

    return epochs_, trange

