import os
import re
from typing import Union
import yaml
import pickle
from pathlib import Path
import pandas as pd

import numpy as np
from pynwb import NWBHDF5IO

import matplotlib.pyplot as plt

from lites2p.dcnv import preprocess
from scipy.stats import ttest_rel as t_test
from statsmodels.stats.multitest import multipletests


def _during(t, a, b):
    return np.argwhere((t <= b) & (t > a)).flatten()


def _mk_epochs(mod, onset, t_bound=(-20, 79), bl_bound=(-700, 0), bl_normalize=False):
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
        baseline = epochs_[:, _during(trange, *bl_bound), :].mean(axis=1, keepdims=True)
        epochs_ = epochs_ - baseline

    return epochs_, trange


def _tone_evoked_p_value(time, dff, evt, baseline_time_range=(-700, 0), activity_time_range=(0, 700)):
    baseline = dff[:, _during(time, *baseline_time_range)].mean(1)
    activity = dff[:, _during(time, *activity_time_range)].mean(1)
    tone_p_values = []
    for tone in np.unique(evt):
        _, tone_p = t_test(activity[evt == tone], baseline[evt == tone])
        tone_p_values.append(tone_p)
    return tone_p_values


def fn_select_roi(_epochs, _trial_info, _time_range):
    _indices, _min_corrected_p_value = [], []
    for i_roi in range(_epochs.shape[2]):
        _data = _epochs[..., i_roi]
        p_values = _tone_evoked_p_value(_time_range, _data, _trial_info,
                                        baseline_time_range=[-800, 0],
                                        activity_time_range=[200, 1000])
        _fdr_corrected_p_values = multipletests(pvals=p_values, method='fdr_bh', alpha=0.05)
        _min_corrected_p_value.append(min(_fdr_corrected_p_values[1]))
        if any(_fdr_corrected_p_values[0]):
            _indices.append(i_roi)
    return np.array(_indices), np.array(_min_corrected_p_value)


class SessionData:
    def __init__(self):
        self._io = None
        self._pm = {}
        self._trial_data = {}
        self._time_range = None
        self._responsive_unit_indices = {}
        self._min_fdr_corrected_p_value = {}
        self._event_related_evoked_response = {}
        self._trial_info = None
        self._pure_index = []
        self._path = None
        self.modules = []
        self._which = None

    def from_path(self, file_path):
        self._io = NWBHDF5IO(os.path.join(file_path, 'nwb-file.nwb'), "r")
        self._path = file_path
        self._load_data()
        self._which = "all-all"
        return self

    def _load_data(self):
        is_str, ps_str = 'ImageSegmentation', 'PlaneSegmentation'
        nwb = self._io.read()

        for md in nwb.modules:
            self._pm[md] = nwb.modules[md].data_interfaces[is_str].plane_segmentations[ps_str].columns[0][:]
            self._pm[md] = np.asarray(self._pm[md], dtype=object)

        self._trial_info = nwb.intervals['trials'].to_dataframe()
        self._pure_index = self._trial_info['type'] == 'pure-tone'
        self.modules = list(nwb.modules.keys())

        for md in self.modules:
            self._trial_data[md], self._time_range = _mk_epochs(nwb.modules[md],
                                                                self._trial_info['start_frame'].values,
                                                                bl_normalize=True)
            self._responsive_unit_indices[md], self._min_fdr_corrected_p_value[md] = (
                fn_select_roi(self._trial_data[md][self._pure_index],
                              self._trial_info['label'][self._pure_index].values,
                              self._time_range))

    def change_selection(self, which: str):
        if not isinstance(which, str):
            raise TypeError("'which' must be string")
        if which not in ["all-all", "all-selective", "pure-all", "pure-selective"]:
            raise ValueError("Unrecognized value for which argument")
        self._which = which

    @property
    def pval(self):
        indices = {}
        for module_name in self.modules:
            indices[module_name] = self._min_fdr_corrected_p_value[module_name]
        return indices

    @property
    def pm(self):
        return {key: self._pm[key][self._responsive_unit_indices[key]] for key in self._pm.keys()}

    @property
    def all_pm(self):
        return self._pm

    @property
    def epochs(self):
        if self._which == 'all-all':
            return {key: self._trial_data[key] for key in self._trial_data.keys()}
        elif self._which == 'pure-all':
            return {key: self._trial_data[key][self._pure_index] for key in self._trial_data.keys()}
        elif self._which == 'all-selective':
            return {key: self._trial_data[key][..., self._responsive_unit_indices[key]]
                    for key in self._trial_data.keys()}
        elif self._which == 'pure-selective':
            return {key: self._trial_data[key][self._pure_index][..., self._responsive_unit_indices[key]]
                    for key in self._trial_data.keys()}
        else:
            raise ValueError(f'Unrecognized value for which argument')

    @property
    def all_roi_epochs(self):
        return {key: self._trial_data[key][self._pure_index] for key in self._trial_data.keys()}

    @property
    def events(self):
        if self._which.startswith("all"):
            return self._trial_info
        elif self._which.startswith("pure"):
            return self._trial_info.loc[self._pure_index, :]
        else:
            raise ValueError(f'Unrecognized value for which argument')

    @property
    def trange(self):
        return self._time_range

    @property
    def tone_evoked(self):
        response = {module_name: [] for module_name in self.modules}
        for module_name in self.modules:
            for tone in self.pure_tones:
                response[module_name].append(self._trial_data[module_name][self._trial_info['label'] == tone].mean(0)
                                             [:, self._responsive_unit_indices[module_name]])
        return {key: np.array(response[key]) for key in response.keys()}

    @property
    def pure_tones(self):
        return np.unique(self._trial_info['label'][self._pure_index])

    def during(self, a, b):
        return np.argwhere((self._time_range <= b) & (self._time_range > a)).flatten()

    def __enter__(self):
        print("Instance of data loader class is acquired")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Instance of data loader class is released")
        if self._io is not None:
            self._io.close()

    def __del__(self):
        if self._io is not None:
            self._io.close()


def get_plane_numbers(suite2p_directory: Union[str, Path]) -> list[int]:
    """
    inputs the suite2p directory, returns the plane numbers
    :param suite2p_directory: string or pathlib.Path, pointing to a directory containing plane<x> subdirectories
    :return: a list of integers
    """
    if isinstance(suite2p_directory, Path):
        suite2p_directory = str(suite2p_directory)
    pattern = re.compile(r'^plane(\d+)$')
    plane_numbers = []
    for item in os.listdir(suite2p_directory):
        if os.path.isdir(os.path.join(suite2p_directory, item)):
            match = pattern.match(item)
            if match:
                plane_numbers.append(int(match.group(1)))
    return plane_numbers


class SessionDataV2:
    def __init__(self, session_path, protocol_path, accepted_only=True):
        print(f"Loading session folder {session_path}")
        if not isinstance(session_path, Path):
            session_path = Path(session_path)
        path = session_path.joinpath("suite2p", "plane0")

        self._is_cell = np.load(path.joinpath("iscell.npy"), allow_pickle=True)
        self._is_axon = np.load(path.joinpath("channel2", "iscell.npy"), allow_pickle=True)
        self._stat_chan1 = np.load(path.joinpath("stat.npy"), allow_pickle=True)
        self._stat_chan2 = np.load(path.joinpath("channel2", "stat.npy"), allow_pickle=True)
        self._f_chan1 = np.load(path.joinpath("F.npy"), allow_pickle=True).T
        self._f_chan2 = np.load(path.joinpath("channel2", "F.npy"), allow_pickle=True).T
        self._ops = np.load(path.joinpath("ops.npy"), allow_pickle=True).item()
        self._composite = plt.imread(session_path.joinpath("images", "composite.png"))

        if accepted_only:
            is_cell = self._is_cell[:, 0].astype(bool).flatten()
            is_axon = self._is_axon[:, 0].astype(bool).flatten()

            self._stat_chan1 = self._stat_chan1[is_cell]
            self._stat_chan2 = self._stat_chan2[is_axon]
            self._f_chan1 = self._f_chan1[..., is_cell]
            self._f_chan2 = self._f_chan2[..., is_axon]

        self._derive_trial_info(protocol_path)
        self._epoch_the_data()

    def _derive_trial_info(self, protocol_path):
        print(f"Protocol description {protocol_path}")
        with open(protocol_path, 'r') as file:
            stimulus_info_protocol = yaml.safe_load(file)

        f0 = stimulus_info_protocol['initial_frame']
        n_tone = len(stimulus_info_protocol['stim_seq'])
        n_repetition = stimulus_info_protocol['nrep']
        trial_duration = stimulus_info_protocol['trial_duration']
        self._tone_onsets = (f0 + np.arange(0, n_tone * n_repetition) * trial_duration).flatten()
        self._tone_labels = np.tile(stimulus_info_protocol['stim_seq'], n_repetition)
        print(f"\tnum repetition in protocol: {n_repetition}")
        print(f"\tnum trial in protocol: {n_repetition * n_tone}")

    def _epoch_the_data(self):
        sampling_rate = self._ops['fs']
        frame_range = np.arange(-20, 79)
        self._trange = frame_range / sampling_rate * 1000
        _in_epoch_indices = (self._tone_onsets[:, np.newaxis] + frame_range).astype('int')

        _data = preprocess(F=self._f_chan1, baseline='minimax', win_baseline=60,
                           sig_baseline=3, fs=sampling_rate, prctile_baseline=8)
        _epochs = _data[_in_epoch_indices, :]
        _baseline = _epochs[:, _during(self._trange, -1000, 0), :].mean(axis=1, keepdims=True)
        self._epochs_chan1 = _epochs - _baseline

        _data = preprocess(F=self._f_chan2, baseline='minimax', win_baseline=60,
                           sig_baseline=3, fs=sampling_rate, prctile_baseline=8)
        _epochs = _data[_in_epoch_indices, :]
        _baseline = _epochs[:, _during(self._trange, -1000, 0), :].mean(axis=1, keepdims=True)
        self._epochs_chan2 = _epochs - _baseline

    @property
    def mean_image(self):
        return {"cell": self._ops['meanImg'], "axon": self._ops['meanImg_chan2'], "both": self._composite.copy()}

    @property
    def roi_masks(self):
        return {"cell": [[s['xpix'], s['ypix']] for s in self._stat_chan1],
                "axon": [[s['xpix'], s['ypix']] for s in self._stat_chan2]}

    @property
    def epochs(self):
        x_cell = self._epochs_chan1.copy()
        x_cell -= x_cell[:, self._trange < 0].mean(axis=1, keepdims=True)

        x_axon = self._epochs_chan2.copy()
        x_axon -= x_axon[:, self._trange < 0].mean(axis=1, keepdims=True)

        return {"cell": x_cell, "axon": x_axon}

    @property
    def tones(self):
        return self._tone_labels

    @property
    def lx(self):
        return self._ops['Lx']

    @property
    def ly(self):
        return self._ops['Ly']

    @property
    def stat(self):
        return {"cell": self._stat_chan1, "axon": self._stat_chan2}

    def between(self, a, b):
        return (self._trange <= b) & (self._trange > a)


def read_master_behavior_file(file_path):
    header = ["session",
              "trial_number",
              "tone",
              "response",
              "no_lick_period",
              "response_time",
              "delay_after_response",
              "total_trial_time_minus_resp_time",
              "lick_frame",
              "reward_frame",
              "total_trial_time",
              "tone_frame",
              "context"]
    df = pd.read_csv(file_path, sep=',', header=None, names=header)
    replacement_map = {1: "H", 2: "M", 3: "FA", 4: "CR"}
    df["response"] = df["response"].map(replacement_map)
    df.replace({'lick_frame': 1000000, 'reward_frame': 1000000}, np.nan, inplace=True)
    return df


def _load_single_channel_from_single_plane(path, accepted_only=True):
    is_accepted = np.load(path.joinpath("iscell.npy"), allow_pickle=True)[:, 0].astype(bool).flatten()
    if not accepted_only:
        is_accepted = np.ones(is_accepted.shape, dtype=bool)

    stat = np.load(path.joinpath("stat.npy"), allow_pickle=True)
    f = np.load(path.joinpath("F.npy"), allow_pickle=True).T
    # f -= 0.7 * np.load(path.joinpath("Fneu.npy"), allow_pickle=True).T
    ops = np.load(path.joinpath("ops.npy"), allow_pickle=True).item()
    stat = stat[is_accepted]
    fl = f[..., is_accepted]
    return ops, stat, fl


def has_all_suite2p_files(path):
    for filename in ["iscell.npy", "stat.npy", "F.npy", "Fneu.npy", "ops.npy"]:
        if not path.joinpath(filename).is_file():
            return False
    return True


class ImagingData:
    def __init__(self, session_path, accepted_only=True):
        print(f"Loading session folder {session_path}")
        if not isinstance(session_path, Path):
            session_path = Path(session_path)

        self._IN_TRIAL_PRE_ONSET_FRAMES = 20

        self._ops = []
        self._composite = []

        self._f_chan1 = []
        self._f_chan2 = []

        self._stat_chan1 = []
        self._stat_chan2 = []

        self._event_frames = {}

        self._plane_numbers = get_plane_numbers(session_path.joinpath("suite2p"))
        if has_all_suite2p_files(session_path.joinpath("suite2p", f"plane{self._plane_numbers[0]}", "channel2")):
            self._has_channel2 = True
        else:
            self._has_channel2 = False

        for plane_number in self._plane_numbers:
            path = session_path.joinpath("suite2p", f"plane{plane_number}")
            self._load_plane(path, accepted_only)

        behavior_filename = session_path.joinpath("master_behavior_file.txt")
        self._derive_trial_info(behavior_filename)

    def _derive_trial_info(self, behavior_filename):
        df = read_master_behavior_file(behavior_filename)
        self._event_frames['tone_onset'] = np.array(df["tone_frame"]) // 2
        self._event_frames['reward'] = np.array(df["reward_frame"]) // 2
        self._event_frames['lick'] = np.array(df["lick_frame"]) // 2
        self._tone_labels = np.array(df["tone"])
        self._trial_type = df["response"].to_numpy()

    def _load_plane(self, plane_directory: Path, accepted_only):
        if not has_all_suite2p_files(plane_directory):
            raise FileNotFoundError(f"{plane_directory} does not contain all suite2p files")
        if not plane_directory.joinpath("images", "composite.png").is_file():
            raise FileNotFoundError(f"{plane_directory} does not contain all suite2p files")
        ops, stat, f = _load_single_channel_from_single_plane(plane_directory, accepted_only)
        composite = plt.imread(plane_directory.joinpath("images", "composite.png"))
        self._ops.append(ops)
        self._composite.append(composite)
        self._stat_chan1.append(stat)
        self._f_chan1.append(f)

        if self._has_channel2:
            if not has_all_suite2p_files(plane_directory):
                raise FileNotFoundError(f"{plane_directory} does not contain all suite2p files")
            channel2_directory = plane_directory.joinpath("channel2")
            ops, stat, f = _load_single_channel_from_single_plane(channel2_directory, accepted_only)
            self._stat_chan2.append(stat)
            self._f_chan2.append(f)

    @property
    def mean_image(self):
        if self._has_channel2:
            return [{"cell": self._ops[p]['meanImg'], "axon": self._ops[p]['meanImg_chan2']}
                    for p in range(len(self._plane_numbers))]
        else:
            return [{"cell": self._ops[p]['meanImg']} for p in range(len(self._plane_numbers))]

    @property
    def roi_masks(self):
        if self._has_channel2:
            return [{"cell": [[s['xpix'], s['ypix']] for s in self._stat_chan1[p]],
                     "axon": [[s['xpix'], s['ypix']] for s in self._stat_chan2[p]]}
                    for p in range(len(self._plane_numbers))]
        else:
            return [{"cell": [[s['xpix'], s['ypix']] for s in self._stat_chan1[p]]}
                    for p in range(len(self._plane_numbers))]

    def epochs(self, frame_anchor, frame_range, baseline_range=None):
        if isinstance(frame_anchor, str):
            frame_points = self._event_frames[frame_anchor]
        elif isinstance(frame_anchor, np.ndarray):
            raise NotImplementedError("")
        else:
            raise TypeError("frame_anchor must be either a hashable entry or a numpy array")

        sampling_rate = self._ops[0]['fs']
        frame_range = np.arange(frame_range[0], frame_range[1])
        na_indices = np.isnan(frame_points)
        frame_points = frame_points[~na_indices]
        in_epoch_indices = (frame_points[:, np.newaxis] + frame_range).astype('int')
        trange = frame_range / sampling_rate * 1000

        out = []
        for i_plane, plane in enumerate(self._plane_numbers):
            temp = {}

            data = preprocess(F=self._f_chan1[i_plane], baseline='minimax', win_baseline=60,
                              sig_baseline=3, fs=sampling_rate, prctile_baseline=8)
            epochs = data[in_epoch_indices, :]
            if baseline_range is not None:
                baseline = epochs[:, (frame_range <= baseline_range[1]) & (frame_range > baseline_range[0]), :].mean(
                    axis=1, keepdims=True)
            else:
                baseline = 0
            temp["cell"] = epochs - baseline

            if self._has_channel2:
                data = preprocess(F=self._f_chan2[i_plane], baseline='minimax', win_baseline=60,
                                  sig_baseline=3, fs=sampling_rate, prctile_baseline=8)
                epochs = data[in_epoch_indices, :]
                if baseline_range is not None:
                    baseline = epochs[:, (frame_range <= baseline_range[1]) & (frame_range > baseline_range[0]), :].mean(
                        axis=1, keepdims=True)
                else:
                    baseline = 0
                temp["axon"] = epochs - baseline

            out.append(temp)

        return out, trange, na_indices

    @property
    def tones(self):
        return self._tone_labels

    @property
    def lx(self):
        return self._ops['Lx']

    @property
    def ly(self):
        return self._ops['Ly']

    @property
    def stat(self):
        if self._has_channel2:
            return [{"cell": self._stat_chan1[p], "axon": self._stat_chan2[p]} for p in range(len(self._plane_numbers))]
        else:
            return [{"cell": self._stat_chan1[p]} for p in range(len(self._plane_numbers))]

    # def between(self, a, b):
    #     return (self._trange <= b) & (self._trange > a)

    @property
    def planes(self):
        return self._plane_numbers

    @property
    def behavior_status(self):
        return self._trial_type
