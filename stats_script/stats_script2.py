import pandas as pd
import matplotlib as plt
import numpy as np
from scipy import interpolate

from typing import List, Tuple
import os
import re
import logging
import datetime

import multiprocessing as mp

from RBDtector.input_handling import input_reader
from RBDtector.data_structures import *

import cProfile

SPLINES = True
RATE = 256
FREQ = '3.90625ms'
FLOW = False

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


# def generate_descripive_statistics(dirname='/home/annika/WORK/RBDtector/Profiling_test'):
# def generate_descripive_statistics(dirname='/home/annika/WORK/RBDtector/Non-Coding-Content/EMG/EMGs'):
def generate_descripive_statistics(dirname='/home/annika/WORK/RBDtector/Non-Coding-Content/Testfiles/test_artefact_menge'):

    human_rated_dirs = find_all_human_rated_directories(dirname)
    s1 = pd.Series(name=('General', '', 'Subject'))
    s2 = pd.Series(name=('General', '', 'Artifact-free REM sleep miniepochs'))
    new_order = ['General']
    new_order.extend(SIGNALS_TO_EVALUATE)
    all_raters = ('Rater 1', 'Rater 2', 'RBDtector')
    h1_vs_h2_raters = ('Rater 1', 'Rater 2')
    h1_vs_h2_df = pd.DataFrame(index=generate_multiindex(*h1_vs_h2_raters))\
        .append(s1).append(s2).reindex(new_order, level=0)
    rbdtector_vs_h1_raters = ('RBDtector', 'Rater 1')
    rbdtector_vs_h1 = pd.DataFrame(index=generate_multiindex(*rbdtector_vs_h1_raters))\
        .append(s1).append(s2).reindex(new_order, level=0)
    rbdtector_vs_h2_raters = ('RBDtector', 'Rater 2')
    rbdtector_vs_h2 = pd.DataFrame(index=generate_multiindex(*rbdtector_vs_h2_raters))\
        .append(s1).append(s2).reindex(new_order, level=0)

    for dirtuple in human_rated_dirs:
        # read directory input
        _, annotation_data = input_reader.read_input(dirtuple[1], read_baseline=False, read_edf=False)
        rbdtector_data = pd.read_pickle(os.path.join(dirtuple[1], 'comparison_pickle'))
        print(rbdtector_data)
        # generate dataframe with REM sleep, artifacts and labels per rater
        evaluation_df = generate_evaluation_dataframe(annotation_data, rbdtector_data, all_raters)
        h1_vs_h2_df = fill_in_comparison_data(h1_vs_h2_df, evaluation_df, dirtuple[0], h1_vs_h2_raters)
        # h1_vs_h2_df.to_excel(dirname + '/human_rater_comparison.xlsx')

        rbdtector_vs_h1 = fill_in_comparison_data(rbdtector_vs_h1, evaluation_df, dirtuple[0], rbdtector_vs_h1_raters)
        rbdtector_vs_h2 = fill_in_comparison_data(rbdtector_vs_h2, evaluation_df, dirtuple[0], rbdtector_vs_h2_raters)


    print(evaluation_df)
    h1_vs_h2_df = add_summary_column(h1_vs_h2_df, h1_vs_h2_raters)
    rbdtector_vs_h1 = add_summary_column(rbdtector_vs_h1, rbdtector_vs_h1_raters)
    rbdtector_vs_h2 = add_summary_column(rbdtector_vs_h2, rbdtector_vs_h2_raters)

    # output_filename = os.path.join('output_tables', '{}_human_scoring_table'.format(os.path.basename(dirname)))
    #
    h1_vs_h2_df.to_excel(dirname + '/human_rater_comparison.xlsx')
    rbdtector_vs_h1.to_excel(dirname + '/rbdtector_vs_h1_comparison.xlsx')
    rbdtector_vs_h2.to_excel(dirname + '/rbdtector_vs_h2_comparison.xlsx')


def add_summary_column(output_df, raters):

    signals = SIGNALS_TO_EVALUATE.copy()
    categories = ('tonic', 'phasic', 'any')
    subject = 'Summary'

    idx = pd.IndexSlice
    output_df.loc[idx[:, :, 'Subject'], subject] = subject
    output_df.loc[idx[:, :, :], subject] = output_df.loc[idx[:, :, :], :].sum(axis=1)

    output_df.loc[idx[:, :, 'Subject'], subject] = subject

    for signal in signals:
        for category in categories:

            output_df.loc[(signal, category, raters[0] + ' % pos'), subject] = \
                (output_df.loc[(signal, category, raters[0] + ' abs pos'), subject] * 100) \
                / output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], subject]

            output_df.loc[(signal, category, raters[1] + ' % pos'), subject] = \
                (output_df.loc[(signal, category, raters[1] + ' abs pos'), subject] * 100) \
                / output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], subject]

            r1_abs_neg = output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], subject] \
                         - output_df.loc[(signal, category, raters[0] + ' abs pos'), subject]
            r2_abs_neg = output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], subject] \
                         - output_df.loc[(signal, category, raters[1] + ' abs pos'), subject]
            p_0 = (output_df.loc[(signal, category, 'shared pos'), subject]
                   + output_df.loc[(signal, category, 'shared neg'), subject]) \
                / output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], subject]

            p_c = (
                          (output_df.loc[(signal, category, raters[0] + ' abs pos'), subject]
                           * output_df.loc[(signal, category, raters[1] + ' abs pos'), subject]
                           ) + (r1_abs_neg * r2_abs_neg)
                  ) \
                / (output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], subject] ** 2)

            output_df.loc[(signal, category, 'Cohen\'s Kappa'), subject] = (p_0 - p_c) / (1 - p_c)



    return output_df


def fill_in_comparison_data(output_df, evaluation_df, subject, raters):
    r1 = '_' + raters[0] if raters[0] != 'RBDtector' else ''
    r2 = '_' + raters[1] if raters[1] != 'RBDtector' else ''

    signals = SIGNALS_TO_EVALUATE.copy()
    categories = ('tonic', 'phasic', 'any')

    artifact_free_rem_miniepochs = evaluation_df['artefact_free_rem_sleep_miniepoch'].sum() / (RATE * 3)

    idx = pd.IndexSlice
    output_df.loc[idx[:, :, 'Subject'], subject] = subject
    output_df.loc[idx[:, :, 'Artifact-free REM sleep miniepochs'], subject] = artifact_free_rem_miniepochs

    length = 3
    miniepochs = '_miniepochs'

    for signal in signals:
        for category in categories:

            if category == 'tonic':
                length = 30
                miniepochs = ''
            else:
                length = 3
                miniepochs = '_miniepochs'

            shared_pos = \
                (evaluation_df[signal + r1 + '_' + category + miniepochs]
                 & evaluation_df[signal + r2 + '_' + category + miniepochs])\
                .sum() / (RATE * length)
            output_df.loc[(signal, category, 'shared pos'), subject] = shared_pos

            shared_neg = \
                (
                    (
                        (~evaluation_df[signal + r1 + '_' + category + miniepochs])
                        & (~evaluation_df[signal + r2 + '_' + category + miniepochs])
                    )
                    & evaluation_df['artefact_free_rem_sleep_miniepoch']
                 ).sum() / (RATE * length)
            output_df.loc[(signal, category, 'shared neg'), subject] = shared_neg

            r1_abs_pos = \
                evaluation_df[signal + r1 + '_' + category + miniepochs]\
                .sum() / (RATE * length)
            output_df.loc[(signal, category, raters[0] + ' abs pos'), subject] = r1_abs_pos

            output_df.loc[(signal, category, raters[0] + ' % pos'), subject] = \
                (r1_abs_pos * 100) / artifact_free_rem_miniepochs

            output_df.loc[(signal, category, raters[0] + ' pos only'), subject] = r1_abs_pos - shared_pos

            r2_abs_pos = \
                evaluation_df[signal + r2 + '_' + category + miniepochs]\
                .sum() / (RATE * length)
            output_df.loc[(signal, category, raters[1] + ' abs pos'), subject] = r2_abs_pos

            output_df.loc[(signal, category, raters[1] + ' % pos'), subject] = \
                (r2_abs_pos * 100) / artifact_free_rem_miniepochs

            output_df.loc[(signal, category, raters[1] + ' pos only'), subject] = r2_abs_pos - shared_pos

            r1_abs_neg = artifact_free_rem_miniepochs - r1_abs_pos
            r2_abs_neg = artifact_free_rem_miniepochs - r2_abs_pos
            p_0 = (shared_pos + shared_neg) \
                / artifact_free_rem_miniepochs

            p_c = ((r1_abs_pos * r2_abs_pos) + (r1_abs_neg * r2_abs_neg)) \
                / (artifact_free_rem_miniepochs ** 2)

            output_df.loc[(signal, category, 'Cohen\'s Kappa'), subject] = (p_0 - p_c) / (1 - p_c)

    return output_df


def generate_multiindex(r1, r2):
    factors = (
        SIGNALS_TO_EVALUATE,
        ('tonic', 'phasic', 'any'),
        (
            'shared pos', 'shared neg',
            r1 + ' abs pos', r1 + ' % pos', r1 + ' pos only',
            r2 + ' abs pos', r2 + ' % pos', r2 + ' pos only',
            'Cohen\'s Kappa'
        )
    )
    index = pd.MultiIndex.from_product(factors, names=["Signal", "Category", 'Description'])
    return index


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
    if not subdirectories:
        raise FileNotFoundError('No subdirectory with two human rater files and a sleep profile file found.')

    return subdirectories


def generate_evaluation_dataframe(annotation_data, rbdtector_data, raters):
    """
    Generate dataframe with REM sleep, artifacts and labels per rater and event type
    :param annotation_data:
    :param rbdtector_data:
    :return: pandas Dataframe
    """
    r1 = raters[0]
    r2 = raters[1]
    rbdtector = raters[2]

    signal_names = SIGNALS_TO_EVALUATE.copy()

    # add sleep profile to df
    df = generate_sleep_profile_df(annotation_data)

    # add artefacts to df
    df = add_artefacts_to_df(df, annotation_data)

    # add RBDtector ratings to df
    df = pd.concat([df, rbdtector_data], axis=1)

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

        # add human rating boolean arrays
        df = add_human_rating_for_signal_type_to_df(df, human_rating1, human_rating1_label_dict, signal_type, '_' + r1)
        df = add_human_rating_for_signal_type_to_df(df, human_rating2, human_rating2_label_dict, signal_type, '_' + r2)

        logging.debug(signal_type + ' end')

    df = df.fillna(False)
    print(df.info())

    return df


def add_human_rating_for_signal_type_to_df(df, human_rating, human_rating_label_dict, signal_type, rater):

    # For all event types (tonic, intermediate, phasic, artefact)
    for event_type in EVENT_TYPE.keys():

        # Create column for human rating of event type
        df[signal_type + rater + '_' + event_type] = pd.Series(False, index=df.index)
        logging.debug(HUMAN_RATING_LABEL[signal_type] + EVENT_TYPE[event_type])

        # Get relevant annotations for column
        human_event_type_indices = \
            human_rating_label_dict.get(HUMAN_RATING_LABEL[signal_type] + EVENT_TYPE[event_type], [])

        # Set bool column true in all rows with annotated indices
        for idx in human_event_type_indices:
            df.loc[
                human_rating.iloc[idx]['event_onset']:human_rating.iloc[idx]['event_end_time'],
                [signal_type + rater + '_' + event_type]
            ] = True

    # adaptions to tonic
    df[signal_type + rater + '_tonic_activity'] = \
        df[signal_type + rater + '_tonic'] & df['artefact_free_rem_sleep_miniepoch']

    tonic_in_30s_epoch = df[signal_type + rater + '_tonic_activity'].squeeze() \
        .resample('30s') \
        .sum() \
        .gt(0)
    df[signal_type + rater + '_tonic'] = tonic_in_30s_epoch
    df[signal_type + rater + '_tonic'] = df[
        signal_type + rater + '_tonic'].ffill()

    # adaptions to phasic
    df[signal_type + rater + '_phasic'] = \
        df[signal_type + rater + '_phasic'] & df['artefact_free_rem_sleep_miniepoch']

    phasic_in_3s_miniepoch = df[signal_type + rater + '_phasic'].squeeze() \
        .resample('3s') \
        .sum() \
        .gt(0)
    df[signal_type + rater + '_phasic_miniepochs'] = phasic_in_3s_miniepoch
    df[signal_type + rater + '_phasic_miniepochs'] = df[signal_type + rater + '_phasic_miniepochs'].ffill()

    # adaptions to any
    df[signal_type + rater + '_any'] = df[signal_type + rater + '_tonic'] | \
                                                   df[signal_type + rater + '_intermediate'] | \
                                                   df[signal_type + rater + '_phasic']

    any_in_3s_miniepoch = df[signal_type + rater + '_any'].squeeze() \
        .resample('3s') \
        .sum() \
        .gt(0)
    df[signal_type + rater + '_any_miniepochs'] = any_in_3s_miniepoch
    df[signal_type + rater + '_any_miniepochs'] = df[signal_type + rater + '_any_miniepochs'].ffill()

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


def generate_sleep_profile_df(annotation_data) -> pd.DataFrame:
    sleep_profile: pd.DataFrame = annotation_data.sleep_profile[1]

    # append a final row to sleep profile with "Artifact" sleep phase
    # for easier concatenation with df while correctly accounting for last 30s sleep profile interval
    sleep_profile = sleep_profile.append(pd.DataFrame({'sleep_phase': 'A'},
                                                      index=[sleep_profile.index.max() + pd.Timedelta('30s')])
                                         )
    sleep_profile.sort_index(inplace=True)

    # resample sleep profile from 30s intervals to 256 Hz, fill all entries with the correct sleeping phase
    # and add it as column to dataframe
    resampled_sleep_profile = sleep_profile.resample(FREQ).ffill()
    # df = pd.concat([df, resampled_sleep_profile], axis=1, join='inner')

    start_datetime = datetime.datetime.strptime(annotation_data.sleep_profile[0]['Start Time'], '%d.%m.%Y %H:%M:%S')
    end_datetime = resampled_sleep_profile.index.max()
    idx = pd.date_range(start_datetime, end_datetime, freq=FREQ)
    df = pd.DataFrame(index=idx)
    df['is_REM'] = resampled_sleep_profile['sleep_phase'] == "REM"
    return df


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


if __name__ == "__main__":
    logging.basicConfig(
        filename='logfile.txt',
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='### %d.%m.%Y %I:%M:%S ###  '
    )
    logging.info('\n----------- START -----------')

    generate_descripive_statistics()
