import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime

from typing import Tuple, Dict, List
import logging

from util.definitions import HUMAN_RATING_LABEL, EVENT_TYPE, definitions_as_string
from util.settings import Settings


def write_output(output_path, human_rating: Tuple[Dict[str, str], pd.DataFrame] = None,
                 calculated_data: pd.DataFrame = None,
                 signal_names: List[str] = None):
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

        # df = calculated_data.copy()
        # df_out = pd.DataFrame()
        # df_out['rater'] = pd.Series('RBDtector')
        # df_out['rem_sleep_duration_in_s'] = pd.Series(df['is_REM'].sum() / 256)
        # df_out['artifact_free_rem_sleep_in_s'] = pd.Series(df['artifact_free_rem_sleep_miniepoch'].sum() / 256)
        # matched_human_miniepochs = {}
        #
        # for signal in signal_names:
        #     start_of_one_signal = datetime.now()
        #     matched_human_miniepochs[signal] = (df[signal + '_human_phasic_miniepochs'] & df[signal + '_phasic_miniepochs'])\
        #                                            .sum() / (256 * 3)
        #
        #     for category in ['tonic', 'phasic', 'any']:
        #         for rater in ['']:
        #             if category == 'tonic':
        #                 df_out['{}_{}{}_in_seconds'.format(signal, category, rater)] = pd.Series(
        #                     df[signal + '{}_{}'.format(rater, category)].sum() / 256
        #                 )
        #
        #                 df_out['{}_{}{}_in_number_of_epochs'.format(signal, category, rater)] = pd.Series(
        #                     df_out['{}_{}{}_in_seconds'.format(signal, category, rater)] / 30
        #                 )
        #
        #             else:
        #                 df_out['{}_{}{}_in_number_of_epochs'.format(signal, category, rater)] = pd.Series(
        #                     df[signal + '{}_{}_miniepochs'.format(rater, category)].sum() / (256 * 3)
        #                 )
        #
        # with open(os.path.join(output_path, 'matched_human_phasic_miniepochs.csv'), 'w') as f:
        #     w = csv.DictWriter(f, matched_human_miniepochs.keys())
        #     w.writeheader()
        #     w.writerow(matched_human_miniepochs)
        #
        # df_out.info()
        # df_out.to_csv(os.path.join(output_path, 'csv_stats_output.csv'), index=False)

        with open(os.path.join(output_path, 'current_settings.csv'), 'w') as f:
            f.write(f"Date: {datetime.now()}"
                    f"{Settings.to_string()}"
                    f"{definitions_as_string()}"
                    )

    except BaseException as e:
        with open(os.path.join(output_path, 'current_settings.csv'), 'w') as f:
            f.write(f"Error in last execution at {datetime.now()}. All current output files are invalid.\n"
                    f"Occurred error: {e}")
        raise e


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
