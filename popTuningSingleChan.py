from pathlib import Path
import os
import numpy as np 
import yaml
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as scipy

def during(t, a, b):
        return np.argwhere((t <= b) & (t > a)).flatten()

class TuningDataSingleChannel:
    def __init__(self, session_path, protocol_path, accepted_only=True):

        print(f"Loading session folder {session_path}")
        if not isinstance(session_path, Path):
            session_path = Path(session_path)
        path = session_path.joinpath("suite2p", "plane0","channel1")

        #self._is_valid = np.load(path.joinpath("iscell.npy"), allow_pickle=True).astype(bool).flatten()
        self._is_valid = np.load(path.joinpath("iscell.npy"), allow_pickle=True)[:, 0].astype(bool).flatten()
        self._stat_chan1 = np.load(path.joinpath("stat.npy"), allow_pickle=True)
        self._f_chan1 = np.load(path.joinpath("F.npy"), allow_pickle=True).T
        self._ops = np.load(path.joinpath("ops.npy"), allow_pickle=True).item()

        if accepted_only:
            self._stat_chan1 = self._stat_chan1[self._is_valid]
            self._f_chan1 = self._f_chan1[..., self._is_valid]

        print(f"\tnum frames in data: {self._f_chan1.shape}")

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
        frame_range = np.arange(-20, 29)
        #frame_range = np.arange(-20, 79)
        self._trange = frame_range / sampling_rate * 1000
        _in_epoch_indices = (self._tone_onsets[:, np.newaxis] + frame_range).astype('int')

        # _data = preprocess(F=self._f_chan1, baseline='minimax', win_baseline=60,
        #                    sig_baseline=3, fs=sampling_rate, prctile_baseline=8)
        _data = self._f_chan1
        _epochs = _data[_in_epoch_indices, :]
        _baseline = _epochs[:, during(self._trange, -1000, 0), :].mean(axis=1, keepdims=True)
        self._epochs_chan1 = _epochs

    @property
    def mean_image(self):
        return {"axon": self._ops['meanImg'], "both": self._ops['meanImg']}

    @property
    def roi_masks(self):
        return {"axon": [[s['xpix'], s['ypix']] for s in self._stat_chan1]}

    @property
    def epochs(self):
        x_axon = self._epochs_chan1.copy()

        return {"axon": x_axon}

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
        return {"axon": self._stat_chan1}

    def between(self, a, b):
        return (self._trange <= b) & (self._trange > a)


def population_average_summary(d, save_path):
    for chan in ["axon"]:
        epochs = d.epochs[chan]        
        std = epochs[:, data.between(-1000, 0)].std(1, keepdims=True)
        epochs -= epochs[:, data.between(-1000, 0)].mean(1, keepdims=True)
        epochs = epochs / std

        fig, axes = plt.subplots(4, 5, figsize=(16, 8))
        unique_stimuli = np.unique(d.tones)

        for ax, s in zip(axes.flat, unique_stimuli):
            ax.set_title(f"{s/1000:0.1f} kHz")
            plot_data = np.nanmean(epochs[d.tones == s], 0).T
            ax.plot(np.nanmean(plot_data, 0))

        for ax in axes.flat:
            ax.grid(False)

        for ax in axes.flat[len(unique_stimuli):]:
            ax.axis("off")
            
        desired_times = [-1000, 0, 1000, 2000, 3000, 4000]
        xtick_positions = [np.abs(data._trange - dt).argmin() for dt in desired_times]
        
        for ax in axes.flat:
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(desired_times, rotation=45)
            ax.set_xlabel("time [ms]")

        fig.suptitle(f"{chan}", fontsize=20)
        fig.tight_layout()
        sns.despine()
        fig.savefig(save_path.joinpath(f"{chan}_population__average_summary.png"))
        plt.close(fig)


plt.plot(data.epochs['axon'].mean((0, 2)))


figure_directory = Path(r"O:\sjk\DATA\imagingData\meso\sk206\007\Figures")
os.makedirs(figure_directory, exist_ok=True)
data = TuningDataSingleChannel(r"O:\sjk\DATA\imagingData\meso\sk206\007", r"W:\su\CODE\klab-pipeline\stimulus_info_v4.yaml", accepted_only=False)
# If you set accepted_only=False in the data loader it will ignore the iscell variable
population_summary(data, figure_directory)



