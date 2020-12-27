import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List
import csv


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
    if human_rating:
        df = human_rating[1]
        onset = pd.to_datetime(df['event_onset'], unit='ms').dt.time
        onset = onset.astype(str)
        onset = onset.apply(lambda x: x[:-3])
        duration_in_seconds = (df['event_end_time'] - df['event_onset']) / np.timedelta64(1, 's')
        df_for_edfBrowser = pd.DataFrame(pd.concat([onset, duration_in_seconds, df['event']], axis=1))
        df_for_edfBrowser.columns = ['event_onset', 'duration', 'event']
        df_for_edfBrowser.to_csv(os.path.join(output_path, 'csv_annotations_for_edfBrowser.txt'), index=False)

        df.to_csv(os.path.join(output_path, 'csv_output.txt'), index=False)

    df = calculated_data
    print(df.info())
    df_out = pd.DataFrame()
    df_out['rater'] = pd.Series('RBDtector')
    df_out['rem_sleep_duration_in_s'] = pd.Series(df['is_REM'].sum() / 256)
    df['artifact_free_rem_sleep_in_s'] = pd.Series(df['artefact_free_rem_sleep_miniepoch'].sum() / 256)
    matched_human_miniepochs = {}

    for signal in signal_names:
        matched_human_miniepochs[signal] = (df[signal + '_human_phasic_miniepochs'] & df[signal + '_phasic_miniepochs'])\
                                               .sum() / (256 * 3)

        for category in ['tonic', 'phasic', 'any']:
            for rater in ['']:
                if category == 'tonic':
                    df_out['{}_{}{}_in_seconds'.format(signal, category, rater)] = pd.Series(
                        df[signal + '{}_{}'.format(rater, category)].sum() / 256
                    )
                    #
                    # df_out['{}_{}{}_in_pct_REM_sleep'.format(signal, category, rater)] = pd.Series(
                    #     (df_out['{}_{}{}_in_seconds'.format(signal, category, rater)]
                    #      / df_out['rem_sleep_duration_in_s'])
                    #     * 100
                    # )

                    df_out['{}_{}{}_in_number_of_epochs'.format(signal, category, rater)] = pd.Series(
                        df_out['{}_{}{}_in_seconds'.format(signal, category, rater)] / 30
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

                    df_out['{}_{}{}_in_number_of_epochs'.format(signal, category, rater)] = pd.Series(
                        df[signal + '{}_{}_miniepochs'.format(rater, category)].sum() / (256 * 3)
                    )

                    # df_out['{}_{}{}_in_epoch_seconds_in_pct_REM_sleep'.format(signal, category, rater)] = pd.Series(
                    #     ((df_out['{}_{}{}_in_number_of_epochs'.format(signal, category, rater)] * 3)
                    #         / df_out['rem_sleep_duration_in_s'])
                    #     * 100
                    # )

    with open(os.path.join(output_path, 'matched_human_phasic_miniepochs.csv'), 'w') as f:
        w = csv.DictWriter(f, matched_human_miniepochs.keys())
        w.writeheader()
        w.writerow(matched_human_miniepochs)

    df_out.info()
    df_out.to_csv(os.path.join(output_path, 'csv_stats_output.csv'), index=False)

    # SAME FOR HUMAN TODO: Create nicer output!
    df_out = pd.DataFrame()
    df_out['rater'] = pd.Series('Human')
    df_out['rem_sleep_duration_in_s'] = pd.Series(df['is_REM'].sum() / 256)
    df['artifact_free_rem_sleep_in_s'] = pd.Series(df['artefact_free_rem_sleep_miniepoch'].sum() / 256)
    for signal in signal_names:
        for category in ['tonic', 'phasic', 'any']:
            for rater in ['_human']:
                if category == 'tonic':
                    df_out['{}_{}{}_in_seconds'.format(signal, category, '')] = pd.Series(
                        df[signal + '{}_{}'.format(rater, category)].sum() / 256
                    )
                    #
                    # df_out['{}_{}{}_in_pct_REM_sleep'.format(signal, category, rater)] = pd.Series(
                    #     (df_out['{}_{}{}_in_seconds'.format(signal, category, rater)]
                    #      / df_out['rem_sleep_duration_in_s'])
                    #     * 100
                    # )

                    df_out['{}_{}{}_in_number_of_epochs'.format(signal, category, '')] = pd.Series(
                        df_out['{}_{}{}_in_seconds'.format(signal, category, '')] / 30
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

                    df_out['{}_{}{}_in_number_of_epochs'.format(signal, category, '')] = pd.Series(
                        df[signal + '{}_{}_miniepochs'.format(rater, category)].sum() / (256 * 3)
                    )

                    # df_out['{}_{}{}_in_epoch_seconds_in_pct_REM_sleep'.format(signal, category, rater)] = pd.Series(
                    #     ((df_out['{}_{}{}_in_number_of_epochs'.format(signal, category, rater)] * 3)
                    #      / df_out['rem_sleep_duration_in_s'])
                    #     * 100
                    # )


    df_out.info()
    df_out.to_csv(os.path.join(output_path, 'csv_human_stats_output.csv'), index=False)

