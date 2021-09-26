import os
import logging
from datetime import datetime

import pandas as pd

import stats_script
from RBDtector.util.stats_definitions import FILE_FINDER, SIGNALS_TO_EVALUATE, stats_definitions_as_string
from RBDtector.util.definitions import EVENT_TYPE, HUMAN_RATING_LABEL, definitions_as_string
from app_logic.PSG import PSG
from input_handling import input_reader
from util.settings import Settings
from util.stats_settings import StatsSettings


def generate_signal_artifact_detection_statistics(dirname='/media/SharedData/EMG/EMGs'):

    # Get a list of all directory paths, that contain human ratings
    human_rated_dirs = stats_script.find_all_human_rated_directories(dirname)

    # Define rater names
    all_raters = ('Rater 1', 'Rater 2', 'RBDtector')
    h1_vs_h2_raters = ('Rater 1', 'Rater 2')
    rbdtector_vs_h1_raters = ('RBDtector', 'Rater 1')
    rbdtector_vs_h2_raters = ('RBDtector', 'Rater 2')

    # Prepare headers for output dataframes
    table_header1 = pd.Series(name=('General', 'Subject'), dtype='str')
    table_header2 = pd.Series(name=('General', 'Global Artifact-free REM sleep miniepochs'), dtype='int')

    # Prepare order of first column of multiindex
    new_order = ['General']
    new_order.extend(SIGNALS_TO_EVALUATE)

    # Build multiindex dataframe for each comparison
    h1_vs_h2_df = pd.DataFrame(index=create_multiindex_for_artifact_comparison(*h1_vs_h2_raters))\
        .append(table_header1).append(table_header2).reindex(new_order, level=0)
    rbdtector_vs_h1 = pd.DataFrame(index=create_multiindex_for_artifact_comparison(*rbdtector_vs_h1_raters))\
        .append(table_header1).append(table_header2).reindex(new_order, level=0)
    rbdtector_vs_h2 = pd.DataFrame(index=create_multiindex_for_artifact_comparison(*rbdtector_vs_h2_raters))\
        .append(table_header1).append(table_header2).reindex(new_order, level=0)

    # Loop over all human-rated PSG directories, calculate its artifact data and add it as column to each dataframe
    for human_rated_dir in human_rated_dirs:

        # read input text files of the directory into AnnotationData object annotation_data
        _, annotation_data = input_reader.read_input(human_rated_dir, read_baseline=False, read_edf=False)

        # read RBDtector dataframe from comparison_pickle in directory
        comparison_pickles = [f.path for f in os.scandir(human_rated_dir)
                              if 'comparison_pickle' in f.name]
        if len(comparison_pickles) == 1:
            comparison_pickle = comparison_pickles[0]
        else:
            print(
                f"No unambiguous comparison_pickle found in {human_rated_dir}. No evaluation possible for this directory.")
            continue

        rbdtector_data = pd.read_pickle(comparison_pickle)

        # generate dataframe with REM sleep, artifacts and labels per rater
        evaluation_df = generate_artifact_evaluation_dataframe(annotation_data, rbdtector_data, all_raters)

        # fill in the current directories data into the comparison dataframes
        h1_vs_h2_df = fill_in_comparison_data(h1_vs_h2_df, evaluation_df, os.path.basename(human_rated_dir),
                                              h1_vs_h2_raters)

        # TODO: include RBDtector signal artifact evaluations as soon as they are available
        # rbdtector_vs_h1 = fill_in_comparison_data(rbdtector_vs_h1, evaluation_df, os.path.basename(human_rated_dir),
        #                                           rbdtector_vs_h1_raters)
        # rbdtector_vs_h2 = fill_in_comparison_data(rbdtector_vs_h2, evaluation_df, os.path.basename(human_rated_dir),
        #                                           rbdtector_vs_h2_raters)
        #


     # Add summary column to statistical comparison dataframes
    h1_vs_h2_df = add_summary_column(h1_vs_h2_df, h1_vs_h2_raters)
    # rbdtector_vs_h1 = add_summary_column(rbdtector_vs_h1, rbdtector_vs_h1_raters)
    # rbdtector_vs_h2 = add_summary_column(rbdtector_vs_h2, rbdtector_vs_h2_raters)

    # Write comparison dataframes to excel files
    h1_vs_h2_df.to_excel(dirname + f'/human_rater_artifact_comparison_{datetime.now()}.xlsx')
    # rbdtector_vs_h1.to_excel(dirname + f'/rbdtector_vs_h1_artifact_comparison_{datetime.now()}.xlsx')
    # rbdtector_vs_h2.to_excel(dirname + f'/rbdtector_vs_h2_artifact_comparison_{datetime.now()}.xlsx')

    # Write last settings and definitions to a text file
    with open(os.path.join(dirname, f'settings_and_definitions_{datetime.now()}'), 'w') as f:
        f.write(Settings.to_string())
        f.write(StatsSettings.to_string())
        f.write(definitions_as_string())
        f.write(stats_definitions_as_string())


def add_summary_column(output_df, raters):


    # prepare definitions and index slice for multiindexing
    signals = SIGNALS_TO_EVALUATE.copy()
    summary_column_name = 'Summary'
    idx = pd.IndexSlice

    # create provisional summary column filled with the simple sum of all rows
    output_df.loc[idx[:, 'Subject'], summary_column_name] = summary_column_name
    output_df.loc[idx[:, :], summary_column_name] = output_df.loc[idx[:, :], :].sum(axis=1)
    output_df.loc[idx[:, 'Subject'], summary_column_name] = summary_column_name

    # loop over all dataframe rows
    for signal in signals:

        total_epoch_count = \
            output_df.loc[idx[:, 'Global Artifact-free REM sleep miniepochs'], summary_column_name].sum()

        output_df.loc[(signal, raters[0] + ' % artifact'), summary_column_name] = \
            (output_df.loc[(signal, raters[0] + ' abs artifact'), summary_column_name] * 100) \
            / total_epoch_count

        output_df.loc[(signal, raters[1] + ' % artifact'), summary_column_name] = \
            (output_df.loc[(signal, raters[1] + ' abs artifact'), summary_column_name] * 100) \
            / total_epoch_count

        r1_abs_neg = total_epoch_count \
                     - output_df.loc[(signal, raters[0] + ' abs artifact'), summary_column_name]
        r2_abs_neg = total_epoch_count \
                     - output_df.loc[(signal, raters[1] + ' abs artifact'), summary_column_name]
        p_0 = (output_df.loc[(signal, 'shared artifact'), summary_column_name]
               + output_df.loc[(signal, 'shared non-artifact'), summary_column_name]) \
            / total_epoch_count

        p_c = (
                      (output_df.loc[(signal, raters[0] + ' abs artifact'), summary_column_name]
                       * output_df.loc[(signal, raters[1] + ' abs artifact'), summary_column_name]
                       ) + (r1_abs_neg * r2_abs_neg)
              ) \
            / (total_epoch_count ** 2)

        output_df.loc[(signal, 'Cohen\'s Kappa'), summary_column_name] = (p_0 - p_c) / (1 - p_c)

    return output_df



def fill_in_comparison_data(output_df, evaluation_df, subject, raters):
    r1 = '_' + raters[0] if raters[0] != 'RBDtector' else ''
    r2 = '_' + raters[1] if raters[1] != 'RBDtector' else ''

    signals = SIGNALS_TO_EVALUATE.copy()

    artifact_free_rem_miniepochs = evaluation_df['artifact_free_rem_sleep_miniepoch'].sum() / (Settings.RATE * 3)

    idx = pd.IndexSlice
    output_df.loc[idx[:, 'Subject'], subject] = subject
    output_df.loc[idx[:, 'Global Artifact-free REM sleep miniepochs'], subject] = artifact_free_rem_miniepochs

    epoch_length = 3
    signal_artifact = '_signal_artifact'

    for signal in signals:

        # global artifacts
        artifact_free_column = evaluation_df['artifact_free_rem_sleep_miniepoch']
        epoch_count = artifact_free_rem_miniepochs

        shared_pos = \
            (evaluation_df[signal + r1 + signal_artifact]
             & evaluation_df[signal + r2 + signal_artifact]) \
                .sum() / (Settings.RATE * epoch_length)
        output_df.loc[(signal, 'shared artifact'), subject] = shared_pos

        shared_neg = \
            (
                    (
                            (~evaluation_df[signal + r1 + signal_artifact])
                            & (~evaluation_df[signal + r2 + signal_artifact])
                    ) & artifact_free_column
            ).sum() / (Settings.RATE * epoch_length)
        output_df.loc[(signal, 'shared non-artifact'), subject] = shared_neg

        r1_abs_pos = \
            evaluation_df[signal + r1 + signal_artifact].sum() / (Settings.RATE * epoch_length)

        output_df.loc[(signal, raters[0] + ' abs artifact'), subject] = r1_abs_pos

        output_df.loc[(signal, raters[0] + ' % artifact'), subject] = \
            (r1_abs_pos * 100) / epoch_count

        output_df.loc[(signal, raters[0] + ' artifact only'), subject] = r1_abs_pos - shared_pos

        r2_abs_pos = \
            evaluation_df[signal + r2 + signal_artifact] \
                .sum() / (Settings.RATE * epoch_length)
        output_df.loc[(signal, raters[1] + ' abs artifact'), subject] = r2_abs_pos

        output_df.loc[(signal, raters[1] + ' % artifact'), subject] = \
            (r2_abs_pos * 100) / epoch_count

        output_df.loc[(signal, raters[1] + ' artifact only'), subject] = r2_abs_pos - shared_pos

        r1_abs_neg = epoch_count - r1_abs_pos
        r2_abs_neg = epoch_count - r2_abs_pos
        p_0 = (shared_pos + shared_neg) \
              / epoch_count

        p_c = ((r1_abs_pos * r2_abs_pos) + (r1_abs_neg * r2_abs_neg)) \
              / (epoch_count ** 2)

        output_df.loc[(signal, 'Cohen\'s Kappa'), subject] = (p_0 - p_c) / (1 - p_c)

    return output_df


def generate_artifact_evaluation_dataframe(annotation_data, rbdtector_data, raters):
    """
    Generate dataframe with REM sleep, global artifacts and signal artifacts per rater
    :param annotation_data:
    :param rbdtector_data:
    :return: pandas Dataframe
    """
    r1 = raters[0]
    r2 = raters[1]
    rbdtector = raters[2]

    signal_names = SIGNALS_TO_EVALUATE.copy()

    # generate sleep profile DataFrame with DatetimeIndex
    df = stats_script.generate_sleep_profile_df(annotation_data)

    # add artifacts to df
    df = stats_script.add_global_artifacts_to_df(df, annotation_data)

    # add RBDtector ratings to df
    df = pd.concat([df, rbdtector_data], axis=1)

    # find all (mini-)epochs of artifact-free REM sleep
    df['artifact_free_rem_sleep_epoch'], df['artifact_free_rem_sleep_miniepoch'] = \
        PSG.find_artifact_free_REM_sleep_epochs_and_miniepochs(df.index, df['is_artifact'], df['is_REM'])

    # find human artifacts
    human_signal_artifacts = PSG.prepare_human_signal_artifacts(annotation_data, df.index, signal_names)

    # add human artifact miniepochs to df
    df_help = pd.DataFrame(index=df.index)
    for signal_name in signal_names:
        for rater_names in [('_human1_artifact', '_' + r1), ('_human2_artifact', '_' + r2)]:
            artifact_signal_series = human_signal_artifacts[signal_name + rater_names[0]]
            artifact_in_3s_miniepoch = artifact_signal_series \
                .resample('3s') \
                .sum() \
                .gt(0)
            df_help['miniepoch_contains_artifact'] = artifact_in_3s_miniepoch
            df_help['miniepoch_contains_artifact'] = df_help['miniepoch_contains_artifact'].ffill()
            df[signal_name + rater_names[1] + '_signal_artifact'] = df_help['miniepoch_contains_artifact'] \
                                                              & df['artifact_free_rem_sleep_miniepoch']

    # Todo: Add RBDtector signal artifacts to evaluation dataframe when available

    df = df.fillna(False)
    print(df.info())

    return df


def create_multiindex_for_artifact_comparison(r1, r2):
    factors = (
        SIGNALS_TO_EVALUATE,
        (
            'shared artifact', 'shared non-artifact',
            r1 + ' abs artifact', r1 + ' % artifact', r1 + ' artifact only',
            r2 + ' abs artifact', r2 + ' % artifact', r2 + ' artifact only',
            'Cohen\'s Kappa'
        )
    )
    index = pd.MultiIndex.from_product(factors, names=["Signal", 'Description'])
    return index



if __name__ == "__main__":
    logging.basicConfig(
        filename='logfile.txt',
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='### %d.%m.%Y %I:%M:%S ###  '
    )
    logging.info('\n----------- START -----------')

    generate_signal_artifact_detection_statistics()