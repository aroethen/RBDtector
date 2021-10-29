from itertools import chain, combinations

import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime

from typing import Tuple, Dict, List
import logging

from util.definitions import HUMAN_RATING_LABEL, EVENT_TYPE, definitions_as_string
from util.settings import Settings


def write_output(output_path,
                 subject_name,
                 calculated_data: pd.DataFrame = None,
                 signal_names: List[str] = None,
                 amplitudes_and_durations: Dict = None
                 ):
    """
    Writes calculated annotations and human rater annotations into csv and xlsx tables for further evaluation 
    and for displaying the respective annotations with the third-party application EDFBrowser.
    :param signal_names: signals to find in calculated data
    :param output_path: Valid path to create output files in
    :param calculated_data: Dataframe of calculated annotations
    :param human_rating: Dataframe of human rater annotations
    """
    logging.info(f'Writing output to {output_path}')
    try:
        write_exact_events_csv(calculated_data, output_path, signal_names)

        cols = calculated_data.columns
        miniepoch_column_names = []
        for signal in signal_names:
            miniepoch_column_names.append('{}_tonic'.format(signal))
            for category in ['phasic', 'any']:
                miniepoch_column_names.append('{}_{}_miniepochs'.format(signal, category))

        try:
            comparison_df = calculated_data[miniepoch_column_names]
            comparison_df.to_pickle(os.path.join(output_path, 'comparison_pickle'))
        except KeyError:
            logging.info("No calculated data exists. This may be due to no viable REM sleep phases in all channels.")

        df_out = create_result_df(calculated_data, signal_names, subject_name, amplitudes_and_durations)

        df_out.transpose().to_excel(os.path.join(output_path, f'RBDtector_results_{datetime.now()}.xlsx'))

        with open(os.path.join(output_path, 'current_settings.csv'), 'w') as f:
            f.write(f"Date: {datetime.now()}"
                    f"{Settings.to_string()}"
                    f"{definitions_as_string()}"
                    )

        #
        #
        # df_channel_combinations = create_channel_combinations_df(calculated_data, signal_names, subject_name)

        return df_out

    except BaseException as e:
        with open(os.path.join(output_path, 'current_settings.csv'), 'w') as f:
            f.write(f"Error in last execution at {datetime.now()}. All current output files are invalid.\n"
                    f"Occurred error: {e}")
        raise e



# def create_channel_combinations_df(calculated_data, signal_names, subject_name):
#     df = calculated_data.copy()
#
#     # create powerset of channel combinations
#     qualities = {'ChinTonic', 'ChinPhasic', 'ChinAny',
#                  'ArmsTonic', 'ArmsPhasic', 'ArmsAny',
#                  'LegsTonic', 'LegsPhasic', 'LegsAny'
#                  }
#     s = list(qualities)
#     all_combinations = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
#



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
        multiindex.get_level_values(0).map(HUMAN_RATING_LABEL) + '_' + multiindex.get_level_values(1)])

    # create result df
    df_out = pd.DataFrame(index=multiindex) \
        .append([table_header1, table_header2, table_header3, table_header4, table_header5]) \
        .reindex(new_order, level=0)

    # fill in results
    idx = pd.IndexSlice

    df_out.loc[idx[:, 'Subject ID'], subject_name] = subject_name

    df_out.loc[idx['Global', 'Global_REM_MiniEpochs'], subject_name] = df['is_REM'].sum() / (Settings.RATE * 3)
    df_out.loc[idx['Global', 'Global_REM_MacroEpochs'], subject_name] = df['is_REM'].sum() / (Settings.RATE * 30)
    df_out.loc[idx['Global', 'Global_REM_MiniEpochs_WO-Artifacts'], subject_name] = \
        df['artifact_free_rem_sleep_miniepoch'].sum() / (Settings.RATE * 3)
    df_out.loc[idx['Global', 'Global_REM_MacroEpochs_WO-Artifacts'], subject_name] = \
        df['artifact_free_rem_sleep_epoch'].sum() / (Settings.RATE * 30)

    for signal_name in signal_names:

        # check existence of calculations
        if not signal_name + '_phasic_miniepochs' in df.columns:
            continue

        df_out.loc[idx[signal_name, HUMAN_RATING_LABEL[signal_name] + '_REM_MiniEpochs_WO-Artifacts'],
                   subject_name] = df[signal_name + '_artifact_free_rem_sleep_miniepoch'].sum() \
                                   / (Settings.RATE * 3)

        df_out.loc[idx[signal_name, HUMAN_RATING_LABEL[signal_name] + '_REM_MacroEpochs_WO-Artifacts'],
                   subject_name] = df[signal_name + '_artifact_free_rem_sleep_epoch'].sum() \
                                   / (Settings.RATE * 30)

        for category in ['tonic', 'phasic', 'any']:

            if category == 'tonic':
                epoch_length = 30
                miniepochs = ''
                epoch_count = \
                    df_out.loc[idx[signal_name, HUMAN_RATING_LABEL[signal_name] + '_REM_MacroEpochs_WO-Artifacts'],
                               subject_name]
            else:
                epoch_length = 3
                miniepochs = '_miniepochs'
                epoch_count = \
                    df_out.loc[idx[signal_name, HUMAN_RATING_LABEL[signal_name] + '_REM_MiniEpochs_WO-Artifacts'],
                               subject_name]

            df_out.loc[idx[signal_name, HUMAN_RATING_LABEL[signal_name] + '_' + category + '_Abs'],
                       subject_name] = df[signal_name + '_' + category + miniepochs].sum() \
                                       / (Settings.RATE * epoch_length)
            abs_count = \
                df_out.loc[idx[signal_name, HUMAN_RATING_LABEL[signal_name] + '_' + category + '_Abs'],
                subject_name]

            df_out.loc[idx[signal_name, HUMAN_RATING_LABEL[signal_name] + '_' + category + '_%'],
                       subject_name] = (abs_count * 100) / epoch_count

            if category == 'phasic':
                df_out.loc[idx[signal_name, HUMAN_RATING_LABEL[signal_name] + '_phasic_Max-Mean-Ampli'],
                           subject_name] = amplitudes_and_durations[signal_name]['phasic']['max_mean']

                df_out.loc[idx[signal_name, HUMAN_RATING_LABEL[signal_name] + '_phasic_Average-Duration'],
                           subject_name] = amplitudes_and_durations[signal_name]['phasic']['mean_duration']

            elif category == 'any':
                df_out.loc[idx[signal_name, HUMAN_RATING_LABEL[signal_name] + '_non-tonic_Max-Mean-Ampli'],
                           subject_name] = amplitudes_and_durations[signal_name]['non-tonic']['max_mean']

                df_out.loc[idx[signal_name, HUMAN_RATING_LABEL[signal_name] + '_non-tonic_Average-Duration'],
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
                list(zip(start_times, end_times, [HUMAN_RATING_LABEL[signal] + EVENT_TYPE[cat]] * len(start_times))))
    rbdtector_events.sort(key=(lambda tpl: tpl[0]))
    with open(
            os.path.join(output_path, 'RBDtection_Events_{}.csv'.format(os.path.basename(output_path))),
            'w'
    ) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(rbdtector_events)
