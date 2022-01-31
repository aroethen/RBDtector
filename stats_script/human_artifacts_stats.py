import os
import logging
from datetime import datetime

import pandas as pd

import stats_script
from RBDtector.util.stats_definitions import FILE_FINDER, SIGNALS_TO_EVALUATE, stats_definitions_as_string
from RBDtector.util.definitions import EVENT_TYPE, HUMAN_RATING_LABEL, definitions_as_string
from app_logic.PSG import PSG
from input_handling import input_reader
from util import settings
from util.stats_settings import StatsSettings


def generate_signal_artifact_detection_statistics(dirname='/home/annika/WORK/RBDtector/Non-Coding-Content/EMGs'):

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
                              if 'comparison_pickle' in Path(f.name, 'RBDtector output')]
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
        f.write(settings.settings_as_string())
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
