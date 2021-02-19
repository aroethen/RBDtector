import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List
import csv

from datetime import datetime

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
    # if human_rating:
    #     df = human_rating[1]
    #     onset = pd.to_datetime(df['event_onset'], unit='ms').dt.time
    #     onset = onset.astype(str)
    #     onset = onset.apply(lambda x: x[:-3])
    #     duration_in_seconds = (df['event_end_time'] - df['event_onset']) / np.timedelta64(1, 's')
    #     df_for_edfBrowser = pd.DataFrame(pd.concat([onset, duration_in_seconds, df['event']], axis=1))
    #     df_for_edfBrowser.columns = ['event_onset', 'duration', 'event']
    #     df_for_edfBrowser.to_csv(os.path.join(output_path, 'csv_annotations_for_edfBrowser.txt'), index=False)
    #
    #     df.to_csv(os.path.join(output_path, 'csv_output.txt'), index=False)

    write_exact_events_csv(calculated_data, output_path, signal_names)

    cols = calculated_data.columns
    miniepoch_column_names = []
    for signal in signal_names:
        miniepoch_column_names.append('{}_tonic'.format(signal))
        for category in ['phasic', 'any']:
            miniepoch_column_names.append('{}_{}_miniepochs'.format(signal, category))

    comparison_df = calculated_data[miniepoch_column_names]
    comparison_df.to_pickle(os.path.join(output_path, 'comparison_pickle'))

    start = datetime.now()
    print('Begin output:' + str(start))
    df = calculated_data
    df_out = pd.DataFrame()
    df_out['rater'] = pd.Series('RBDtector')
    a = datetime.now()
    print('A: ', a - start)
    df_out['rem_sleep_duration_in_s'] = pd.Series(df['is_REM'].sum() / 256)
    b = datetime.now()
    print('B: ', b - a)
    df_out['artifact_free_rem_sleep_in_s'] = pd.Series(df['artefact_free_rem_sleep_miniepoch'].sum() / 256)
    matched_human_miniepochs = {}
    c = datetime.now()
    print('C: ', c - b)

    before_signal_loop = datetime.now()
    print('Before signal loop:' + str(before_signal_loop))

    for signal in signal_names:
        start_of_one_signal = datetime.now()
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
        one_signal_duration = datetime.now() - start_of_one_signal
        print('One signal duration:' + str(one_signal_duration))

    with open(os.path.join(output_path, 'matched_human_phasic_miniepochs.csv'), 'w') as f:
        w = csv.DictWriter(f, matched_human_miniepochs.keys())
        w.writeheader()
        w.writerow(matched_human_miniepochs)

    df_out.info()
    df_out.to_csv(os.path.join(output_path, 'csv_stats_output.csv'), index=False)


def write_exact_events_csv(calculated_data, output_path, signal_names):
    rbdtector_events = []
    for signal in signal_names:
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
                diff = intermediate_ratings.astype(int).diff()
                cat = 'intermediate'
            else:
                diff = calculated_data[signal + '_' + category].astype(int).diff()

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
