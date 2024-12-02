import os
import json
import pickle
import argparse
import numpy as np

from uuid import uuid4
from datetime import datetime
from zoneinfo import ZoneInfo

from pynwb import NWBHDF5IO, NWBFile

from pynwb.file import Subject
from pynwb.ophys import (
    Fluorescence,
    ImageSegmentation,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries
)

from sbxreader import sbx_memmap


def parse_events(p):
    f0 = p['initial-frame']
    trial_duration = p['trial-duration']
    n_repetition = p['n-repetition']
    tones = np.array(p['tone-sequence'])
    n_tone = tones.size

    start_time = (f0 + np.arange(0, n_tone * n_repetition) * trial_duration).flatten()
    end_time = start_time + trial_duration
    event_labels = np.tile(tones, n_repetition)
    event_types = np.full(event_labels.shape, "pure-tone")
    event_types[event_labels == 64001] = "white-noise"
    event_types[event_labels == 64002] = "upward-sweep"
    event_types[event_labels == 64003] = "downward-sweep"

    return start_time.astype(float), end_time.astype(float), event_labels, event_types


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-json', type=str, required=True, help='path to json file')
    parser.add_argument('-tmp-dir', type=str, required=False, default="", help='temporary directory')
    args = parser.parse_args()
    json_file_name = args.json
    with open(json_file_name, 'r') as fp:
        _params = json.load(fp)
    _dirs = _params['directories']
    if args.tmp_dir != "":
        _dirs['tmp'] = args.tmp_dir
    return _params, _dirs


if __name__ == '__main__':
    file_path = "/Volumes/Andrew's External Hard Drive/Data/sk83/session0/raw.sk83.003.001/sk83_003_001.sbx"
    json_path = "/Users/mohrabi/PycharmProjects/thalcor-tuning/meta/sk132.full.json"
    file_extension = os.path.split(file_path)[-1].split('.')[-1]

    # if file_extension == 'sbx':
    #
    # else:
    #     raise NotImplementedError()

    # params, dirs = parse_arguments()
    with open(json_path, 'r') as fp:
        params = json.load(fp)
    # imaging_rate = params['sampling-rate']

    subject = Subject(
        subject_id=params['subject']['subject-id'],
        description=params['subject']['description'],
        age=params['subject']['age'],
        genotype=params['subject']['genotype'],
        sex=params['subject']['sex'],
        species=params['subject']['species'],
        weight=params['subject']['weight'],
        strain=params['subject']['strain'],
        date_of_birth=datetime.strptime(params['subject']['date-of-birth'],
                                        "%Y-%m-%d").replace(tzinfo=ZoneInfo("EST"))
    )

    nwbfile = NWBFile(
        session_description=params['session-description'],
        identifier=str(uuid4()),
        session_start_time=datetime.strptime(params['session-start-date'], "%Y-%m-%d").replace(tzinfo=ZoneInfo("EST")),
        experimenter=params['experimenter'],
        experiment_description=params["experiment-description"],
        session_id=str(uuid4()),
        institution=params['institution'],
        keywords=params['keywords'],
        protocol=params['protocol'],
        surgery=params['surgery'],
        virus=params['virus'],
        lab=params['lab'],
        epoch_tags=set(),
        subject=subject,
    )

    # trial_start_time, trial_end_time, trial_labels, trial_types = parse_events(params['trials'])
    # nwbfile.add_trial_column(name="start_frame", description="stimulus onset frame")
    # nwbfile.add_trial_column(name="label", description="Stimulus identifier")
    # nwbfile.add_trial_column(name="type", description="Presented tone type")
    # for start, stop, stim, e_type in zip(trial_start_time, trial_end_time, trial_labels, trial_types):
    #     nwbfile.add_trial(start_time=start / imaging_rate,
    #                       stop_time=start / imaging_rate + 0.1,
    #                       start_frame=start,
    #                       label=stim,
    #                       type=e_type)

    for dev_spec in params['devices']:
        nwbfile.create_device(
            name=dev_spec['name'],
            description=dev_spec['description'],
            manufacturer=dev_spec['manufacturer'],
        )

    if "Microscope" not in nwbfile.devices.keys():
        raise KeyError("Microscope device is not found in device list. Check the input json file.")
    microscope = nwbfile.devices["Microscope"]

    if file_extension == "sbx":
        interface = sbx_memmap(file_path)
        meta_data = interface.metadata
        num_planes = meta_data['num_planes']
        num_channels = meta_data['num_channels']
        frame_rate = meta_data['frame_rate']
        frame_size = meta_data['frame_size']

        # for i_channel in range(num_channels):
        #     optical_channel = OpticalChannel(
        #         name=f'channel{i_channel+1}',
        #         description='',
        #         emission_lambda=500.0,
        #     )
        #
        #     for i_plane in range(num_plane):
        #         imaging_plane = nwbfile.create_imaging_plane(
        #             name='cell',
        #             optical_channel=optical_channel,
        #             imaging_rate=frame_rate,
        #             description="",
        #             device=microscope,
        #             excitation_lambda=600.0,
        #             indicator='indicator',
        #             location='location',
        #             grid_spacing=[0.01, 0.01],
        #             grid_spacing_unit="meters",
        #             origin_coords=[1.0, 2.0, 3.0],
        #             origin_coords_unit="meters"
        #         )
        #
        #         two_p_series = TwoPhotonSeries(
        #             name="TwoPhotonSeries",
        #             description="Raw 2p data",
        #             data=interface[],
        #             imaging_plane=imaging_plane,
        #             rate=frame_rate,
        #             unit="normalized amplitude",
        #         )
    # nwbfile.add_acquisition(two_p_series)
    #
    # with NWBHDF5IO(os.path.join(dirs['tmp'], 'nwb-file.nwb'), mode="w") as io:
    #     io.write(nwbfile)
