import pandas as pd
import numpy as np

from typing import List, Tuple
import os
import logging
from datetime import datetime

from RBDtector.input_handling import input_reader

from RBDtector.util.stats_definitions import FILE_FINDER, SIGNALS_TO_EVALUATE
from RBDtector.util.definitions import EVENT_TYPE, HUMAN_RATING_LABEL, definitions_as_string
from RBDtector.util.settings import Settings


# def generate_descripive_statistics(dirname='/home/annika/WORK/RBDtector/Profiling_test'):
# def generate_descripive_statistics(dirname='/home/annika/WORK/RBDtector/Non-Coding-Content/EMG/EMGs'):
def generate_descripive_statistics(dirname='/home/annika/WORK/RBDtector/Non-Coding-Content/Testfiles/test_artifact_menge'):

    # Get a list of all directory paths, that contain human ratings
    human_rated_dirs = find_all_human_rated_directories(dirname)

    # Define rater names
    all_raters = ('Rater 1', 'Rater 2', 'RBDtector')
    h1_vs_h2_raters = ('Rater 1', 'Rater 2')
    rbdtector_vs_h1_raters = ('RBDtector', 'Rater 1')
    rbdtector_vs_h2_raters = ('RBDtector', 'Rater 2')

    # Prepare headers and order for output dataframes
    table_header1 = pd.Series(name=('General', '', 'Subject'))
    table_header2 = pd.Series(name=('General', '', 'Artifact-free REM sleep miniepochs'))
    new_order = ['General']
    new_order.extend(SIGNALS_TO_EVALUATE)

    # Build multiindex dataframe for each comparison
    h1_vs_h2_df = pd.DataFrame(index=create_full_multiindex_for_raters(*h1_vs_h2_raters))\
        .append(table_header1).append(table_header2).reindex(new_order, level=0)
    rbdtector_vs_h1 = pd.DataFrame(index=create_full_multiindex_for_raters(*rbdtector_vs_h1_raters))\
        .append(table_header1).append(table_header2).reindex(new_order, level=0)
    rbdtector_vs_h2 = pd.DataFrame(index=create_full_multiindex_for_raters(*rbdtector_vs_h2_raters))\
        .append(table_header1).append(table_header2).reindex(new_order, level=0)

    # Loop over all human-rated PSG directories, calculate its statistical data and add it as column to each dataframe
    for human_rated_dir in human_rated_dirs:

        # read input text files of the directory into AnnotationData object annotation_data
        _, annotation_data = input_reader.read_input(human_rated_dir, read_baseline=False, read_edf=False)

        # read RBDtector dataframe from comparison_pickle in directory
        comparison_pickles = [f.path for f in os.scandir(human_rated_dir)
                              if 'comparison_pickle' in f.name]
        if len(comparison_pickles) == 1:
            comparison_pickle = comparison_pickles[0]
        else:
            print(f"No unambiguous comparison_pickle found in {human_rated_dir}. No evaluation possible for this directory.")
            continue

        rbdtector_data = pd.read_pickle(comparison_pickle)

        # generate dataframe with REM sleep, artifacts and labels per rater
        evaluation_df = generate_evaluation_dataframe(annotation_data, rbdtector_data, all_raters)

        # fill in the current directories data into the comparison dataframes
        h1_vs_h2_df = fill_in_comparison_data(h1_vs_h2_df, evaluation_df, os.path.basename(human_rated_dir), h1_vs_h2_raters)
        rbdtector_vs_h1 = fill_in_comparison_data(rbdtector_vs_h1, evaluation_df, os.path.basename(human_rated_dir), rbdtector_vs_h1_raters)
        rbdtector_vs_h2 = fill_in_comparison_data(rbdtector_vs_h2, evaluation_df, os.path.basename(human_rated_dir), rbdtector_vs_h2_raters)

    # Add summary column to statistical comparison dataframes
    h1_vs_h2_df = add_summary_column(h1_vs_h2_df, h1_vs_h2_raters)
    rbdtector_vs_h1 = add_summary_column(rbdtector_vs_h1, rbdtector_vs_h1_raters)
    rbdtector_vs_h2 = add_summary_column(rbdtector_vs_h2, rbdtector_vs_h2_raters)

    # Write comparison dataframes to excel files
    h1_vs_h2_df.to_excel(dirname + f'/human_rater_comparison_{datetime.now()}.xlsx')
    rbdtector_vs_h1.to_excel(dirname + f'/rbdtector_vs_h1_comparison_{datetime.now()}.xlsx')
    rbdtector_vs_h2.to_excel(dirname + f'/rbdtector_vs_h2_comparison_{datetime.now()}.xlsx')

    # Write last settings and definitions to a text file
    with open(os.path.join(dirname, f'settings_and_definitions_{datetime.now()}'), 'w') as f:
        f.write(Settings.to_string())
        f.write(definitions_as_string())


def add_summary_column(output_df, raters):

    # prepare definitions and index slice for multiindexing
    signals = SIGNALS_TO_EVALUATE.copy()
    categories = ('tonic', 'phasic', 'any')
    summary_column_name = 'Summary'
    idx = pd.IndexSlice

    # create provisional summary column filled with the simple sum of all rows
    output_df.loc[idx[:, :, 'Subject'], summary_column_name] = summary_column_name
    output_df.loc[idx[:, :, :], summary_column_name] = output_df.loc[idx[:, :, :], :].sum(axis=1)
    output_df.loc[idx[:, :, 'Subject'], summary_column_name] = summary_column_name

    # loop over all dataframe rows
    for signal in signals:
        for category in categories:

            output_df.loc[(signal, category, raters[0] + ' % pos'), summary_column_name] = \
                (output_df.loc[(signal, category, raters[0] + ' abs pos'), summary_column_name] * 100) \
                / output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], summary_column_name]

            output_df.loc[(signal, category, raters[1] + ' % pos'), summary_column_name] = \
                (output_df.loc[(signal, category, raters[1] + ' abs pos'), summary_column_name] * 100) \
                / output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], summary_column_name]

            r1_abs_neg = output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], summary_column_name] \
                         - output_df.loc[(signal, category, raters[0] + ' abs pos'), summary_column_name]
            r2_abs_neg = output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], summary_column_name] \
                         - output_df.loc[(signal, category, raters[1] + ' abs pos'), summary_column_name]
            p_0 = (output_df.loc[(signal, category, 'shared pos'), summary_column_name]
                   + output_df.loc[(signal, category, 'shared neg'), summary_column_name]) \
                / output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], summary_column_name]

            p_c = (
                          (output_df.loc[(signal, category, raters[0] + ' abs pos'), summary_column_name]
                           * output_df.loc[(signal, category, raters[1] + ' abs pos'), summary_column_name]
                           ) + (r1_abs_neg * r2_abs_neg)
                  ) \
                / (output_df.loc[idx['General', '', 'Artifact-free REM sleep miniepochs'], summary_column_name] ** 2)

            output_df.loc[(signal, category, 'Cohen\'s Kappa'), summary_column_name] = (p_0 - p_c) / (1 - p_c)



    return output_df


def fill_in_comparison_data(output_df, evaluation_df, subject, raters):
    r1 = '_' + raters[0] if raters[0] != 'RBDtector' else ''
    r2 = '_' + raters[1] if raters[1] != 'RBDtector' else ''

    signals = SIGNALS_TO_EVALUATE.copy()
    categories = ('tonic', 'phasic', 'any')

    artifact_free_rem_miniepochs = evaluation_df['artifact_free_rem_sleep_miniepoch'].sum() / (Settings.RATE * 3)

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
                .sum() / (Settings.RATE * length)
            output_df.loc[(signal, category, 'shared pos'), subject] = shared_pos

            shared_neg = \
                (
                    (
                        (~evaluation_df[signal + r1 + '_' + category + miniepochs])
                        & (~evaluation_df[signal + r2 + '_' + category + miniepochs])
                    )
                    & evaluation_df['artifact_free_rem_sleep_miniepoch']
                 ).sum() / (Settings.RATE * length)
            output_df.loc[(signal, category, 'shared neg'), subject] = shared_neg

            r1_abs_pos = \
                evaluation_df[signal + r1 + '_' + category + miniepochs]\
                .sum() / (Settings.RATE * length)
            output_df.loc[(signal, category, raters[0] + ' abs pos'), subject] = r1_abs_pos

            output_df.loc[(signal, category, raters[0] + ' % pos'), subject] = \
                (r1_abs_pos * 100) / artifact_free_rem_miniepochs

            output_df.loc[(signal, category, raters[0] + ' pos only'), subject] = r1_abs_pos - shared_pos

            r2_abs_pos = \
                evaluation_df[signal + r2 + '_' + category + miniepochs]\
                .sum() / (Settings.RATE * length)
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


def create_full_multiindex_for_raters(r1, r2):
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


def find_all_human_rated_directories(directory_name) -> List[str]:
    """
    :returns list of all somnography directories containing human ratings and sleep profiles.
                Format: List[absolute directory paths]
    """

    # find all subdirectories of given directory
    subdirectories = [d.path for d in os.scandir(os.path.abspath(directory_name))
                      if d.is_dir()]

    # remove subdirectories without human rater files from subdirectories
    for subdir in subdirectories.copy():
        found_human_rater_file = False
        found_second_human_rater_file = False
        found_sleep_profile = False
        for file in os.scandir(subdir):
            filename = file.name
            if FILE_FINDER['human_rating_2'] in filename:
                found_second_human_rater_file = True
            elif FILE_FINDER['human_rating'] in filename:
                found_human_rater_file = True
            elif FILE_FINDER['sleep_profile'] in filename:
                found_sleep_profile = True

        if (not (found_human_rater_file or found_second_human_rater_file)) or (not found_sleep_profile):
            subdirectories.remove(subdir)

    subdirectories.sort()
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

    # add artifacts to df
    df = add_artifacts_to_df(df, annotation_data)

    # add RBDtector ratings to df
    df = pd.concat([df, rbdtector_data], axis=1)

    # find all 3s miniepochs of artifact-free REM sleep
    artifact_signal = df['is_artifact'].squeeze()
    artifact_in_3s_miniepoch = artifact_signal \
        .resample('3s') \
        .sum()\
        .gt(0)
    df['miniepoch_contains_artifact'] = artifact_in_3s_miniepoch
    df['miniepoch_contains_artifact'] = df['miniepoch_contains_artifact'].ffill()
    df['artifact_free_rem_sleep_miniepoch'] = df['is_REM'] & ~df['miniepoch_contains_artifact']

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
        df = add_human_rating_for_signal_type_to_df(df, [human_rating1, human_rating2], human_rating1_label_dict, signal_type, '_' + r1)
        df = add_human_rating_for_signal_type_to_df(df, [human_rating2, human_rating1], human_rating2_label_dict, signal_type, '_' + r2)

        logging.debug(signal_type + ' end')

    df = df.fillna(False)
    print(df.info())

    return df


def add_human_rating_for_signal_type_to_df(df, human_ratings, human_rating_label_dict, signal_type, rater):

    human_rating = human_ratings[0]

    # For all event types (tonic, intermediate, phasic, artifact)
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

    # Add second human rating artifacts to dataframe if available
    if human_ratings > 1:
        second_rater = human_ratings[1]
        second_rater_rating_label_dict = second_rater.groupby('event').groups

        # Get relevant annotations for column
        second_rater_event_type_indices = \
            second_rater_rating_label_dict.get(HUMAN_RATING_LABEL[signal_type] + EVENT_TYPE['artifact'], [])

        # Set bool column true in all rows with annotated indices
        for idx in second_rater_event_type_indices:
            df.loc[
                second_rater.iloc[idx]['event_onset']:second_rater.iloc[idx]['event_end_time'],
                [signal_type + '_human_artifact']
            ] = True

    # adaptions to tonic
    df[signal_type + rater + '_tonic_activity'] = \
        df[signal_type + rater + '_tonic'] & df['artifact_free_rem_sleep_miniepoch']

    tonic_in_30s_epoch = df[signal_type + rater + '_tonic_activity'].squeeze() \
        .resample('30s') \
        .sum() \
        .gt(0)
    df[signal_type + rater + '_tonic'] = tonic_in_30s_epoch
    df[signal_type + rater + '_tonic'] = df[
        signal_type + rater + '_tonic'].ffill()

    # adaptions to phasic
    df[signal_type + rater + '_phasic'] = \
        df[signal_type + rater + '_phasic'] & df['artifact_free_rem_sleep_miniepoch']

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


def add_artifacts_to_df(df, annotation_data):

    arousals: pd.DataFrame = annotation_data.arousals[1]
    df['artifact_event'] = pd.Series(False, index=df.index)
    for label, on, off in zip(arousals['event'], arousals['event_onset'], arousals['event_end_time']):
        df.loc[on:off, ['artifact_event']] = True

    if Settings.FLOW:
        flow_events = annotation_data.flow_events[1]
        df['flow_event'] = pd.Series(False, index=df.index)
        for label, on, off in zip(flow_events['event'], flow_events['event_onset'], flow_events['event_end_time']):
            df.loc[on:off, ['flow_event']] = True

    # add conditional column 'is_artifact'
    if Settings.FLOW:
        df['is_artifact'] = np.logical_or(df['artifact_event'], df['flow_event'])
    else:
        df['is_artifact'] = df['artifact_event']

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
    resampled_sleep_profile = sleep_profile.resample(Settings.FREQ).ffill()
    # df = pd.concat([df, resampled_sleep_profile], axis=1, join='inner')

    start_datetime = datetime.strptime(annotation_data.sleep_profile[0]['Start Time'], '%d.%m.%Y %H:%M:%S')
    end_datetime = resampled_sleep_profile.index.max()
    idx = pd.date_range(start_datetime, end_datetime, freq=Settings.FREQ)
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
                start_date = datetime.strptime(split_list[1].strip(), '%d.%m.%Y %H').date()
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
