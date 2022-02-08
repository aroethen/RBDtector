import sys
from itertools import chain, combinations

import pandas as pd
import numpy as np
from pathlib import Path
import os
import csv
from datetime import datetime

from typing import Dict, List
import logging

from util.definitions import HUMAN_RATING_LABEL, EVENT_TYPE, definitions_as_string
from util import settings
from util.error_for_display import ErrorForDisplay


def write_output(psg_path,
                 subject_name,
                 calculated_data: pd.DataFrame = None,
                 signal_names: List[str] = None,
                 amplitudes_and_durations: Dict = None
                 ):
    """
    Writes calculated annotations and human rater annotations into csv and xlsx tables for further evaluation 
    and for displaying the respective annotations with the third-party application EDFBrowser.
    :param signal_names: signals to find in calculated data
    :param psg_path: Path of the PSG directory in order to create a directory 'RBDtector output' there
    :param calculated_data: Dataframe of calculated annotations
    :param human_rating: Dataframe of human rater annotations
    """
    logging.info(f'Writing output to {psg_path}')
    try:
        output_path = Path(psg_path, 'RBDtector output')
        output_path.mkdir(exist_ok=True)

    except OSError:
        raise ErrorForDisplay(f'An output directory inside {output_path} could not be created. '
                              f'No specific output possible for this PSG.')

    try:
        write_exact_events_csv(calculated_data, output_path, signal_names)

        miniepoch_column_names = []
        for signal in signal_names:
            miniepoch_column_names.append('{}_tonic'.format(signal))
            for category in ['phasic', 'any']:
                miniepoch_column_names.append('{}_{}_miniepochs'.format(signal, category))

        df_out = create_result_df(calculated_data, signal_names, subject_name, amplitudes_and_durations)

        df_out.transpose().to_excel(os.path.normpath(
            os.path.join(
                output_path,
                f'RBDtector_results_{str(datetime.now()).replace(" ", "_").replace(":", "-")}'
                f'.xlsx')))

        df_channel_combinations = create_channel_combinations_df(calculated_data, subject_name)
        df_channel_combinations.to_excel(os.path.normpath(
            os.path.join(output_path, f'Channel_combinations_{str(datetime.now()).replace(" ", "_").replace(":", "-")}.xlsx')))

        with open(os.path.normpath(os.path.join(output_path, 'current_settings.csv')), 'w') as f:
            f.write(f"Date: {str(datetime.now()).replace(' ', '_').replace(':', '-')}"
                    f"{settings.settings_as_string()}"
                    f"{definitions_as_string()}"
                    )

        print(settings.SIGNALS_TO_EVALUATE)

        return df_out, df_channel_combinations

    except BaseException as e:
        with open(os.path.normpath(os.path.join(output_path, 'current_settings.csv')), 'w') as f:
            f.write(f"Error in last execution at {str(datetime.now()).replace(' ', '_').replace(':', '-')}. "
                    f"All current output files are invalid.\n"
                    f"Occurred error: {e}")
        raise e



def create_channel_combinations_df(calculated_data, subject_name):
    df = calculated_data.copy()

    # combinations of all chin channels, arm channels and leg channels respectively
    basic_combinations = {}

    # names of channel combination basic combinations
    combination_keys = ['ChinTonic', 'ChinPhasic', 'ChinAny',
                        'ArmsTonic', 'ArmsPhasic', 'ArmsAny',
                        'LegsTonic', 'LegsPhasic', 'LegsAny']

    # keys under which to find the channel combinations in the
    combination_keys_in_df = []
    for location in [settings.CHIN, settings.ARMS, settings.LEGS]:
        for event_type in ['_tonic', '_phasic_miniepochs', '_any_miniepochs']:

            keys_for_location_and_event_type = []
            try:
                for elem in location:
                    # construct the actual keys, e.g. 'EMG_phasic_miniepochs'
                    try:
                        keys_for_location_and_event_type.append(settings.SIGNALS_TO_EVALUATE[elem] + event_type)
                    except IndexError:
                        logging.error(f'SIGNALS_TO_EVALUATE ({settings.SIGNALS_TO_EVALUATE}) '
                                      f'does not contain an element at index {elem} as defined as an electrode '
                                      f'placement (CHIN, LEGS, ARMS) inside the configuration file.')
                        raise ErrorForDisplay(
                            f'SIGNALS_TO_EVALUATE ({settings.SIGNALS_TO_EVALUATE}) '
                            f'does not contain an element at index {elem} as defined as an electrode '
                            f'placement (CHIN, LEGS, ARMS) inside the configuration file.')
            except TypeError:
                # if location contains only one element and is not a list
                try:
                    keys_for_location_and_event_type.append(settings.SIGNALS_TO_EVALUATE[location] + event_type)
                except KeyError:
                    continue
            combination_keys_in_df.append(keys_for_location_and_event_type)

    # combine all channels given for CHIN, ARM and LEGS, respectively
    for key, df_keys in zip(combination_keys, combination_keys_in_df):
        try:
            basic_combinations[key] = np.logical_or.reduce(df[df_keys], axis=1)
        except KeyError:
            continue

    # create powerset of channel combinations
    s = list(basic_combinations.keys())
    all_combinations = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))[1:]
    all_combinations_as_string = [','.join(x) for x in all_combinations]

    channel_lists_list = [[basic_combinations[key]
                           for key in combination_list]
                          for combination_list in all_combinations]
    combined_channels_list = [np.logical_or.reduce(channel_combination)
                              for channel_combination in channel_lists_list]
    df_channel_combinations = pd.DataFrame(dict(zip(all_combinations_as_string, combined_channels_list)),
                                           index=df.index)

    # count miniepochs of activity in artifact-free REM sleep per combination
    sum_of_channel_combinations = df_channel_combinations.loc[:, :].sum(axis=0)
    epoch_length = 3
    epoch_count = df['artifact_free_rem_sleep_miniepoch'].sum() / (settings.RATE * epoch_length)
    df_out = pd.DataFrame(sum_of_channel_combinations.values,
                          columns=[subject_name, ],
                          index=all_combinations_as_string).transpose()
    df_out = (df_out / (settings.RATE * epoch_length)) * 100 / epoch_count

    return df_out


def create_result_df(calculated_data, signal_names, subject_name, amplitudes_and_durations):

    df = calculated_data.copy()

    # create Multiindex
    table_header1 = pd.Series(name=('', 'Subject ID'), dtype='str')
    table_header2 = pd.Series(name=('Global', 'Global_REM_MiniEpochs'), dtype='int')
    table_header3 = pd.Series(name=('Global', 'Global_REM_MacroEpochs'), dtype='int')
    table_header4 = pd.Series(name=('Global', 'Global_REM_MiniEpochs_WO-Artifacts'), dtype='int')
    table_header5 = pd.Series(name=('Global', 'Global_REM_MacroEpochs_WO-Artifacts'), dtype='int')

    new_order = ['', 'Global']
    new_order.extend(signal_names)

    factors = (
        signal_names,
        (
            'REM_MiniEpochs_WO-Artifacts', 'REM_MacroEpochs_WO-Artifacts',
            'tonic_Abs', 'tonic_%',
            'phasic_Abs', 'phasic_%',
            'any_Abs', 'any_%',
            'phasic_Max-Mean-Ampli', 'phasic_Average-Duration',
            'non-tonic_Max-Mean-Ampli', 'non-tonic_Average-Duration'
        )
    )

    multiindex = pd.MultiIndex.from_product(factors, names=["Signal", 'Description'])
    multiindex = pd.MultiIndex.from_arrays([
        multiindex.get_level_values(0),
        multiindex.get_level_values(0).map(lambda x: HUMAN_RATING_LABEL.get(x, x)) + '_' + multiindex.get_level_values(1)])

    # create result df
    df_out = pd.DataFrame(index=multiindex) \
        .append([table_header1, table_header2, table_header3, table_header4, table_header5]) \
        .reindex(new_order, level=0)

    # fill in results
    idx = pd.IndexSlice

    df_out.loc[idx[:, 'Subject ID'], subject_name] = subject_name

    df_out.loc[idx['Global', 'Global_REM_MiniEpochs'], subject_name] = df['is_REM'].sum() / (settings.RATE * 3)
    df_out.loc[idx['Global', 'Global_REM_MacroEpochs'], subject_name] = df['is_REM'].sum() / (settings.RATE * 30)
    df_out.loc[idx['Global', 'Global_REM_MiniEpochs_WO-Artifacts'], subject_name] = \
        df['artifact_free_rem_sleep_miniepoch'].sum() / (settings.RATE * 3)
    df_out.loc[idx['Global', 'Global_REM_MacroEpochs_WO-Artifacts'], subject_name] = \
        df['artifact_free_rem_sleep_epoch'].sum() / (settings.RATE * 30)

    for signal_name in signal_names:

        # check existence of calculations
        if not signal_name + '_phasic_miniepochs' in df.columns:
            continue

        df_out.loc[idx[signal_name,
                       HUMAN_RATING_LABEL.get(signal_name, signal_name) + '_REM_MiniEpochs_WO-Artifacts'],
                   subject_name] = df[signal_name + '_artifact_free_rem_sleep_miniepoch'].sum() \
                                   / (settings.RATE * 3)

        df_out.loc[idx[signal_name,
                       HUMAN_RATING_LABEL.get(signal_name, signal_name) + '_REM_MacroEpochs_WO-Artifacts'],
                   subject_name] = df[signal_name + '_artifact_free_rem_sleep_epoch'].sum() \
                                   / (settings.RATE * 30)

        for category in ['tonic', 'phasic', 'any']:

            if category == 'tonic':
                epoch_length = 30
                miniepochs = ''
                epoch_count = \
                    df_out.loc[idx[signal_name, HUMAN_RATING_LABEL.get(signal_name, signal_name) + '_REM_MacroEpochs_WO-Artifacts'],
                               subject_name]
            else:
                epoch_length = 3
                miniepochs = '_miniepochs'
                epoch_count = \
                    df_out.loc[idx[signal_name, HUMAN_RATING_LABEL.get(signal_name, signal_name) + '_REM_MiniEpochs_WO-Artifacts'],
                               subject_name]

            df_out.loc[idx[signal_name, HUMAN_RATING_LABEL.get(signal_name, signal_name) + '_' + category + '_Abs'],
                       subject_name] = df[signal_name + '_' + category + miniepochs].sum() \
                                       / (settings.RATE * epoch_length)
            abs_count = \
                df_out.loc[idx[signal_name, HUMAN_RATING_LABEL.get(signal_name, signal_name) + '_' + category + '_Abs'],
                subject_name]

            df_out.loc[idx[signal_name, HUMAN_RATING_LABEL.get(signal_name, signal_name) + '_' + category + '_%'],
                       subject_name] = (abs_count * 100) / epoch_count

            if category == 'phasic':
                df_out.loc[idx[signal_name, HUMAN_RATING_LABEL.get(signal_name, signal_name) + '_phasic_Max-Mean-Ampli'],
                           subject_name] = amplitudes_and_durations[signal_name]['phasic']['max_mean']

                df_out.loc[idx[signal_name, HUMAN_RATING_LABEL.get(signal_name, signal_name) + '_phasic_Average-Duration'],
                           subject_name] = amplitudes_and_durations[signal_name]['phasic']['mean_duration']

            elif category == 'any':
                df_out.loc[idx[signal_name, HUMAN_RATING_LABEL.get(signal_name, signal_name) + '_non-tonic_Max-Mean-Ampli'],
                           subject_name] = amplitudes_and_durations[signal_name]['non-tonic']['max_mean']

                df_out.loc[idx[signal_name, HUMAN_RATING_LABEL.get(signal_name, signal_name) + '_non-tonic_Average-Duration'],
                           subject_name] = amplitudes_and_durations[signal_name]['non-tonic']['mean_duration']

    return df_out


def write_exact_events_csv(calculated_data, output_path, signal_names):
    rbdtector_events = []
    for signal in signal_names:
        if (not signal + '_phasic' in calculated_data.columns) or (not signal + '_tonic' in calculated_data.columns):
            logging.info(f'No calculations for channel {signal} found. Skipped in output.')
            continue
        for category in ['tonic', 'phasic', 'any']:
            cat = category
            if category == 'any':
                intermediate_ratings = np.logical_and(
                    calculated_data[signal + '_any'],
                    np.logical_not(
                        np.logical_or(
                            calculated_data[signal + '_phasic'],
                            calculated_data[signal + '_tonic']
                        )
                    )
                )
                diff = intermediate_ratings.fillna('-99').astype('int').diff()
                cat = 'intermediate'
            else:
                diff = calculated_data[signal + '_' + category].fillna('-99').astype('int').diff()

            start_times = diff[diff == 1].index.array
            end_times = diff[diff == -1].index.array
            rbdtector_events.extend(
                list(zip(start_times, end_times, [HUMAN_RATING_LABEL.get(signal, signal) + EVENT_TYPE[cat]] * len(start_times))))
    rbdtector_events.sort(key=(lambda tpl: tpl[0]))
    with open(
            os.path.normpath(os.path.join(output_path, 'RBDtection_Events_{}.csv'.format(os.path.basename(output_path)))),
            'w'
    ) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(rbdtector_events)
