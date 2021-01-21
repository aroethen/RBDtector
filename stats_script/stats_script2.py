import pandas as pd
import matplotlib as plt
import numpy as np
from scipy import interpolate

from typing import List, Tuple
import os
import re
import logging
import datetime
from itertools import chain

from RBDtector.input_handling import input_reader
from RBDtector.data_structures import *

SPLINES = True
RATE = 256
FREQ = '3.90625ms'
FLOW = True

FILE_FINDER = {
    'edf': '.edf',
    'sleep_profile': 'Sleep profile',
    'flow_events': 'Flow Events',
    'arousals': 'Classification Arousals',
    'baseline': 'Start-Baseline',
    'human_rating': 'Generic',
    'human_rating_2': 'Generic_NO'
}

QUALITIES = [
    'LeftArmPhasic', 'LeftArmTonic', 'LeftArmAny',
    'RightArmPhasic', 'RightArmTonic', 'RightArmAny',
    'LeftLegPhasic', 'LeftLegTonic', 'LeftLegAny',
    'RightLegPhasic', 'RightLegTonic', 'RightLegAny',
    'ChinPhasic', 'ChinTonic', 'ChinAny',
    'LeftArmArtifact', 'RightArmArtifact', 'LeftLegArtifact', 'RightLegArtifact', 'ChinArtifact'
]

HUMAN_RATING_LABEL = {
    'EMG': 'Chin',
    'PLM l': 'LeftLeg',
    'PLM r': 'RightLeg',
    'AUX': 'RightArm',
    'Akti.': 'LeftArm'
}

EVENT_TYPE = {
    'tonic': 'Tonic',
    'intermediate': 'Any',
    'phasic': 'Phasic',
    'artefact': 'Artifact'
}

SIGNALS_TO_EVALUATE = ['EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.']


def generate_descripive_statistics(dirname='../Non-Coding-Content/EMG'):
    human_rated_dirs = find_all_human_rated_directories(dirname)
    output_df = pd.DataFrame()
    first_dir = True

    for dirtuple in human_rated_dirs:
        # create row and add subject

        # read directory input
        raw_data, annotation_data = input_reader.read_input(dirtuple[1], read_baseline=False)

        # generate dataframe with REM sleep, artifacts and labels per rater
        df = generate_evaluation_dataframe(raw_data, annotation_data)

        # compare and count epochs / miniepochs
        row_df = compare_and_count_epochs(df)
        row_df['Subject'] = dirtuple[0]
        if first_dir:
            output_df = row_df.copy()
            first_dir = False
        else:
            output_df = output_df.append(row_df)


    # columns = ['Subject']
    # quality_columns = list(chain.from_iterable((col + ' [s]', col + ' [% REM]') for col in QUALITIES))
    # columns.extend(quality_columns)

    print(df)
    # output_filename = os.path.join('output_tables', '{}_human_scoring_table'.format(os.path.basename(dirname)))
    #
    output_df.to_csv(dirname + '/human_rater_comparison.csv')
    output_df.to_excel(dirname + '/human_rater_comparison.xlsx')


def find_all_human_rated_directories(directory_name) -> List[Tuple[str, str]]:
    """
    :returns list of Tuples of all somnography directories containing human ratings and sleep profiles.
                Format: List[Tuple[directory name, direcory path
    """

    # find all subdirectories of given directory
    subdirectories = [(d.name, d.path) for d in os.scandir(os.path.abspath(directory_name))
                      if d.is_dir()]

    # remove subdirectories without human rater files from subdirectories
    for subdir in subdirectories.copy():
        found_human_rater_file = False
        found_second_human_rater_file = False
        found_sleep_profile = False
        for file in os.scandir(subdir[1]):
            filename = file.name
            if FILE_FINDER['human_rating_2'] in filename:
                found_second_human_rater_file = True
            elif FILE_FINDER['human_rating'] in filename:
                found_human_rater_file = True
            elif FILE_FINDER['sleep_profile'] in filename:
                found_sleep_profile = True

        if (not (found_human_rater_file or found_second_human_rater_file)) or (not found_sleep_profile):
            subdirectories.remove(subdir)

    subdirectories.sort(key=lambda t: t[0])

    return subdirectories


def generate_evaluation_dataframe(raw_data, annotation_data):
    """
    Generate dataframe with REM sleep, artifacts and labels per rater and event type
    :return: pandas Dataframe
    """
    start_datetime = raw_data.get_header()['startdate']
    signal_names = SIGNALS_TO_EVALUATE.copy()

    # prepare DataFrame with DatetimeIndex
    idx = create_datetime_index(raw_data, start_datetime)
    df = pd.DataFrame(index=idx)

    # add sleep profile to df
    df = add_sleep_profile_to_df(df, annotation_data)

    # add artefacts to df
    df = add_artefacts_to_df(df, annotation_data)

    # cut off all samples before start of sleep_profile assessment
    start_of_first_full_sleep_phase = df['sleep_phase'].ne('A').idxmax()
    df = df[start_of_first_full_sleep_phase:]

    # find all 3s miniepochs of artifact-free REM sleep
    artefact_signal = df['is_artefact'].squeeze()
    artefact_in_3s_miniepoch = artefact_signal \
        .resample('3s') \
        .sum()\
        .gt(0)
    df['miniepoch_contains_artefact'] = artefact_in_3s_miniepoch
    df['miniepoch_contains_artefact'] = df['miniepoch_contains_artefact'].ffill()
    df['artefact_free_rem_sleep_miniepoch'] = df['is_REM'] & ~df['miniepoch_contains_artefact']

    # process human rating for evaluation per signal and event
    human_rating1 = annotation_data.human_rating[0][1]
    human_rating1_label_dict = human_rating1.groupby('event').groups
    logging.debug(human_rating1_label_dict)
    try:
        human_rating2 = annotation_data.human_rating[1][1]
        human_rating2_label_dict = human_rating2.groupby('event').groups
        logging.debug(human_rating2_label_dict)
    except IndexError:
        human_rating2 = None
        human_rating2_label_dict = None

    # FOR EACH EMG SIGNAL:
    for signal_type in signal_names.copy():
        logging.debug(signal_type + ' start')

        # Check if signal type exists in edf file
        try:
            signal_array = raw_data.get_data_channels()[signal_type].get_signal()
        except KeyError:
            signal_names.remove(signal_type)
            continue

        # Resample to 256 Hz
        df[signal_type] = signal_to_256hz_datetimeindexed_series(
            raw_data, signal_array, signal_type, start_datetime
        )[start_of_first_full_sleep_phase:]

        # add human rating boolean arrays
        df = add_human_rating_for_signal_type_to_df(df, human_rating1, human_rating1_label_dict, signal_type, 'rater1')
        df = add_human_rating_for_signal_type_to_df(df, human_rating2, human_rating2_label_dict, signal_type, 'rater2')

        logging.debug(signal_type + ' end')
    print(df.info())

    return df


def add_human_rating_for_signal_type_to_df(df, human_rating, human_rating_label_dict, signal_type, rater):

    # For all event types (tonic, intermediate, phasic, artefact)
    for event_type in EVENT_TYPE.keys():

        # Create column for human rating of event type
        df[signal_type + '_human_' + rater + '_' + event_type] = pd.Series(False, index=df.index)
        logging.debug(HUMAN_RATING_LABEL[signal_type] + EVENT_TYPE[event_type])

        # Get relevant annotations for column
        human_event_type_indices = \
            human_rating_label_dict.get(HUMAN_RATING_LABEL[signal_type] + EVENT_TYPE[event_type], [])

        # Set bool column true in all rows with annotated indices
        for idx in human_event_type_indices:
            df.loc[
                human_rating.iloc[idx]['event_onset']:human_rating.iloc[idx]['event_end_time'],
                [signal_type + '_human_' + rater + '_' + event_type]
            ] = True

    # adaptions to tonic
    df[signal_type + '_human_' + rater + '_tonic'] = \
        df[signal_type + '_human_' + rater + '_tonic'] & df['artefact_free_rem_sleep_miniepoch']

    tonic_in_30s_epoch = df[signal_type + '_human_' + rater + '_tonic'].squeeze() \
        .resample('30s') \
        .sum() \
        .gt(0)
    df[signal_type + '_human_' + rater + '_tonic_epochs'] = tonic_in_30s_epoch
    df[signal_type + '_human_' + rater + '_tonic_epochs'] = df[
        signal_type + '_human_' + rater + '_tonic_epochs'].ffill()

    # adaptions to phasic
    df[signal_type + '_human_' + rater + '_phasic'] = \
        df[signal_type + '_human_' + rater + '_phasic'] & df['artefact_free_rem_sleep_miniepoch']

    phasic_in_3s_miniepoch = df[signal_type + '_human_' + rater + '_phasic'].squeeze() \
        .resample('3s') \
        .sum() \
        .gt(0)
    df[signal_type + '_human_' + rater + '_phasic_miniepochs'] = phasic_in_3s_miniepoch
    df[signal_type + '_human_' + rater + '_phasic_miniepochs'] = df[signal_type + '_human_' + rater + '_phasic_miniepochs'].ffill()

    # adaptions to any
    df[signal_type + '_human_' + rater + '_any'] = df[signal_type + '_human_' + rater + '_tonic'] | \
                                                   df[signal_type + '_human_' + rater + '_intermediate'] | \
                                                   df[signal_type + '_human_' + rater + '_phasic']

    any_in_3s_miniepoch = df[signal_type + '_human_' + rater + '_any'].squeeze() \
        .resample('3s') \
        .sum() \
        .gt(0)
    df[signal_type + '_human_' + rater + '_any_miniepochs'] = any_in_3s_miniepoch
    df[signal_type + '_human_' + rater + '_any_miniepochs'] = df[signal_type + '_human_' + rater + '_any_miniepochs'].ffill()

    return df


def add_artefacts_to_df(df, annotation_data):

    arousals: pd.DataFrame = annotation_data.arousals[1]
    df['artefact_event'] = pd.Series(False, index=df.index)
    for label, on, off in zip(arousals['event'], arousals['event_onset'], arousals['event_end_time']):
        df.loc[on:off, ['artefact_event']] = True

    if FLOW:
        flow_events = annotation_data.flow_events[1]
        df['flow_event'] = pd.Series(False, index=df.index)
        for label, on, off in zip(flow_events['event'], flow_events['event_onset'], flow_events['event_end_time']):
            df.loc[on:off, ['flow_event']] = True

    # add conditional column 'is_artefact'
    if FLOW:
        df['is_artefact'] = np.logical_or(df['artefact_event'], df['flow_event'])
    else:
        df['is_artefact'] = df['artefact_event']

    return df


def add_sleep_profile_to_df(df: pd.DataFrame, annotation_data) -> pd.DataFrame:
    sleep_profile: pd.DataFrame = annotation_data.sleep_profile[1]

    # append a final row to sleep profile with "Artifact" sleep phase
    # for easier concatenation with df while correctly accounting for last 30s sleep profile interval
    sleep_profile = sleep_profile.append(pd.DataFrame({'sleep_phase': 'A'},
                                                      index=[sleep_profile.index.max() + pd.Timedelta('30s')])
                                         )
    sleep_profile.sort_index(inplace=True)

    # resample sleep profile from 2Hz(30s intervals) to 256 Hz, fill all entries with the correct sleeping phase
    # and add it as column to dataframe
    resampled_sleep_profile = sleep_profile.resample(str(1000 / RATE) + 'ms').ffill()
    df = pd.concat([df, resampled_sleep_profile], axis=1, join='inner')
    df['is_REM'] = df['sleep_phase'] == "REM"
    return df


def create_datetime_index(raw_data, start_datetime):
    freq_in_ms = 1000 / RATE
    emg_channel = raw_data.get_data_channels()['EMG']
    sample_rate = emg_channel.get_sample_rate()
    sample_length = len(emg_channel.get_signal())
    index_length = (sample_length / sample_rate) * RATE
    return pd.date_range(start_datetime, freq=str(freq_in_ms) + 'ms', periods=index_length)


def signal_to_256hz_datetimeindexed_series(raw_data, signal_array, signal_type, start_datetime):
    sample_rate = raw_data.get_data_channels()[signal_type].get_sample_rate()
    duration_in_seconds = len(signal_array) / float(sample_rate)
    old_sample_points = np.arange(len(signal_array))
    new_sample_points = np.arange(len(signal_array), step=(sample_rate / RATE), dtype=np.double)

    if SPLINES:
        tck = interpolate.splrep(old_sample_points, signal_array)
        resampled_signal_array = interpolate.splev(new_sample_points, tck)
    else:
        raise NotImplementedError('TODO - Non-Splines-Methode implementieren')
        # TODO: - Non-Splines-Methode implementieren
        # resampled_signal_array = signal.resample_poly(
        #     signal_array, RATE, sample_rate, padtype='constant', cval=float('NaN')
        # )

    # tidx = pd.date_range(start_datetime, freq='3.90625ms', periods=len(old_sample_points))
    # ^ NUR WENN OLD_POINTS mit Sample-Rate 256 aufgenommen wurden

    freq_in_ms = 1000/RATE

    idx = pd.date_range(start_datetime, freq=str(freq_in_ms)+'ms', periods=len(new_sample_points))
    resampled_signal_dtseries = pd.Series(resampled_signal_array, index=idx, name=signal_type)


    # plt.plot(tidx, signal_array)
    # plt.plot(idx, resampled_signal_array, c='gold')
    # plt.show()
    return resampled_signal_dtseries


def find_start_date_in_file(sleep_profile):

    start_date = None

    with open(sleep_profile, 'r', encoding='utf-8') as f:

        text_in_lines = f.readlines()

        for line in text_in_lines:
            split_list = line.split(':')
            if 'Start Time' in split_list[0]:
                start_date = datetime.datetime.strptime(split_list[1].strip(), '%d.%m.%Y %H').date()
                break

    return start_date


def compare_and_count_epochs(df):
    df_out = pd.DataFrame()

    df_out['rem_sleep_duration_in_s'] = pd.Series(df['is_REM'].sum() / 256)
    df['artifact_free_rem_sleep_in_s'] = pd.Series(df['artefact_free_rem_sleep_miniepoch'].sum() / 256)
    shared_phasic_miniepochs = {}
    shared_any_miniepochs = {}
    shared_tonic_epochs = {}

    for signal in SIGNALS_TO_EVALUATE.copy():
        shared_phasic_miniepochs[signal] = (df[signal + '_human_rater1_phasic_miniepochs'] & df[signal + '_human_rater2_phasic_miniepochs'])\
                                               .sum() / (256 * 3)
        shared_any_miniepochs[signal] = (df[signal + '_human_rater1_phasic_miniepochs'] & df[signal + '_human_rater2_phasic_miniepochs'])\
                                               .sum() / (256 * 3)
        shared_tonic_epochs[signal] = {}

        for category in ['tonic', 'phasic', 'any']:
            if category == 'tonic':
                df_out['{}_shared_{}_epochs'.format(signal, category)] = \
                    (
                            df[signal + '_human_rater1_{}_epochs'.format(category)] &
                            df[signal + '_human_rater2_{}_epochs'.format(category)]
                    ).sum() / (256 * 30)
            else:
                df_out['{}_shared_{}_miniepochs'.format(signal, category)] = \
                    (
                            df[signal + '_human_rater1_{}_miniepochs'.format(category)] &
                            df[signal + '_human_rater2_{}_miniepochs'.format(category)]
                    ).sum() / (256 * 3)

            for rater in ['_human_rater1', '_human_rater2']:
                if category == 'tonic':
                    # df_out['{}_{}{}_in_seconds'.format(signal, category, rater)] = pd.Series(
                    #     df[signal + '{}_{}'.format(rater, category)].sum() / 256
                    # )
                    #
                    # df_out['{}_{}{}_in_pct_REM_sleep'.format(signal, category, rater)] = pd.Series(
                    #     (df_out['{}_{}{}_in_seconds'.format(signal, category, rater)]
                    #      / df_out['rem_sleep_duration_in_s'])
                    #     * 100
                    # )

                    df_out['{}_{}{}_epochs'.format(signal, category, rater)] = pd.Series(
                        df[signal + '{}_{}_epochs'.format(rater, category)].sum() / (256 * 30)
                    )

                else:
                    # df_out['{}_{}{}_in_seconds'.format(signal, category, rater)] = pd.Series(
                    #     df[signal + '{}_{}'.format(rater, category)].sum() / 256
                    # )
                    #
                    # df_out['{}_{}{}_in_pct_REM_sleep'.format(signal, category, rater)] = pd.Series(
                    #     (df_out['{}_{}{}_in_seconds'.format(signal, category, rater)]
                    #      / df_out['rem_sleep_duration_in_s'])
                    #     * 100
                    # )

                    df_out['{}_{}{}_epochs'.format(signal, category, rater)] = pd.Series(
                        df[signal + '{}_{}_miniepochs'.format(rater, category)].sum() / (256 * 3)
                    )

                    # df_out['{}_{}{}_in_epoch_seconds_in_pct_REM_sleep'.format(signal, category, rater)] = pd.Series(
                    #     ((df_out['{}_{}{}_in_number_of_epochs'.format(signal, category, rater)] * 3)
                    #         / df_out['rem_sleep_duration_in_s'])
                    #     * 100
                    # )
    return df_out




if __name__ == "__main__":
    logging.basicConfig(
        filename='logfile.txt',
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='### %d.%m.%Y %I:%M:%S ###  '
    )
    logging.info('\n----------- START -----------')

    generate_descripive_statistics()
