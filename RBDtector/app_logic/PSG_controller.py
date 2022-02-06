import logging
import os
import traceback
from datetime import datetime

import pandas as pd

from util.settings import SIGNALS_TO_EVALUATE
from util.definitions import SLEEP_CLASSIFIERS
from util import settings
from app_logic.PSG import PSG
from input_handling import input_reader as ir
from output_handling import csv_writer
from util.error_for_display import ErrorForDisplay


class PSGController:
    """High-level controller for PSG evaluation functionality.
     Provides easy to use functions for RBD event detection."""

    @staticmethod
    def run_rbd_detection(input_path: str, output_path: str):

        psg = PSG(input_path, output_path)


        raw_data, annotation_data = ir.read_input(directory_name=input_path,
                                                  signals_to_load=settings.SIGNALS_TO_EVALUATE.copy(),
                                                  read_human_rating=settings.HUMAN_ARTIFACTS,
                                                  read_baseline=settings.HUMAN_BASELINE)

        df_signals, is_REM_series, is_global_artifact_series, signal_names, sleep_phase_series = \
            psg.prepare_evaluation(raw_data, annotation_data, settings.SIGNALS_TO_EVALUATE.copy(), settings.FLOW)

        # find all (mini)epochs of global artifact-free REM sleep
        is_global_artifact_free_rem_sleep_epoch_series, is_global_artifact_free_rem_sleep_miniepoch_series = \
            psg.find_artifact_free_REM_sleep_epochs_and_miniepochs(
                idx=df_signals.index, artifact_signal_series=is_global_artifact_series, is_REM_series=is_REM_series)


        # FIND SIGNAL ARTIFACTS
        signal_artifacts = pd.DataFrame(index=df_signals.index)
        for signal_name in signal_names:
            signal_artifacts[signal_name + '_signal_artifact'] = 0

        if settings.HUMAN_ARTIFACTS:
            human_signal_artifacts = psg.prepare_human_signal_artifacts(
                 annotation_data, df_signals.index, signal_names)
            signal_artifacts = pd.DataFrame(index=df_signals.index)
            for signal_name in signal_names:
                signal_artifacts[signal_name + '_signal_artifact'] = \
                    human_signal_artifacts[signal_name + '_human_artifact']

        if settings.SNORE:
            snore_series = sleep_phase_series.str.lower() == SLEEP_CLASSIFIERS['SNORE'].lower()
            signal_artifacts[f'{settings.SIGNALS_TO_EVALUATE[0]}_signal_artifact'] = \
                signal_artifacts[f'{settings.SIGNALS_TO_EVALUATE[0]}_signal_artifact'] | snore_series

        artifact_free_rem_sleep_per_signal = psg.find_signal_artifact_free_REM_sleep_epochs_and_miniepochs(
            df_signals.index, is_REM_series, is_global_artifact_series, signal_artifacts,
            signal_names)

        # FIND BASELINE
        if settings.HUMAN_BASELINE:
            df_baselines, _ = psg.find_baselines(df_signals=df_signals, signal_names=signal_names,
                                              use_human_baselines=True, annotation_data=annotation_data)
        else:
            df_baselines, df_baseline_artifacts = psg.find_baselines(df_signals=df_signals, signal_names=signal_names,
                                              use_human_baselines=False, is_rem_series=is_REM_series,
                                              artifact_free_rem_sleep_per_signal=artifact_free_rem_sleep_per_signal,
                                              annotation_data=annotation_data)

            for signal_name in signal_names:
                signal_artifacts[signal_name + '_signal_artifact'] = \
                    signal_artifacts[signal_name + '_signal_artifact'] \
                    | df_baseline_artifacts[signal_name + '_baseline_artifact']

            artifact_free_rem_sleep_per_signal = psg.find_signal_artifact_free_REM_sleep_epochs_and_miniepochs(
                df_signals.index, is_REM_series, is_global_artifact_series, signal_artifacts,
                signal_names)

        rbd_events, amplitudes_and_durations = psg.detect_rbd_events(df_signals=df_signals, df_baselines=df_baselines,
                                           artifact_free_rem_sleep_per_signal=artifact_free_rem_sleep_per_signal,
                                           signal_names=signal_names, annotation_data=annotation_data)


        df_out, df_channel_combinations = csv_writer.write_output(output_path,
                                subject_name=os.path.basename(input_path),
                                calculated_data=pd.concat([rbd_events, is_REM_series,
                                                           is_global_artifact_free_rem_sleep_epoch_series,
                                                           is_global_artifact_free_rem_sleep_miniepoch_series,
                                                           artifact_free_rem_sleep_per_signal],
                                                          axis=1, verify_integrity=True),
                                signal_names=signal_names,
                                amplitudes_and_durations=amplitudes_and_durations)

        return df_out, df_channel_combinations


def single_psg_run(input_path, output_path = None, dev_run: bool = False):
    """Error-controlled evaluation of a single PSG"""

    if output_path is None:
        output_path = input_path

    error_messages = ''

    try:
        PSGController.run_rbd_detection(input_path, output_path)

    except (OSError, ErrorForDisplay) as e:
        if dev_run:
            print(f'Expectable error in input PSG {input_path}:\n {e}')
        if not dev_run:
            error_messages = error_messages + f'Error in input PSG {input_path}:\n {e}\n' \
                                              f'Full error message can be found in log file.\n\n'

        logging.error(f'Error in file {input_path}:\n {e}')
        logging.error(traceback.format_exc())

    except BaseException as e:
        if dev_run:
            print(f'Unexpected error in PSG {input_path}:\n {e}')

        if not dev_run:
            error_messages = error_messages + f'Unexpected error in PSG {input_path}:\n {e}\n' \
                                              f'Full error message in log file. Please contact developer.\n\n'

        logging.error(f'Unexpected error in file {input_path}:\n {e}')
        logging.error(traceback.format_exc())

    return error_messages


def superdir_run(path, dev_run: bool = False):

    dirlist = os.listdir(path)
    reading_problems = []
    df_out_combined = pd.DataFrame()
    df_channel_combinations_combined = pd.DataFrame()

    error_messages = ''
    first = True

    for child in dirlist:
        abs_child = os.path.normpath(os.path.join(path, child))
        if os.path.isdir(abs_child):
            try:
                df_out, df_channel_combinations = PSGController.run_rbd_detection(abs_child, abs_child)

                if first:
                    df_out_combined = df_out.copy()
                    df_channel_combinations_combined = df_channel_combinations.copy()

                    first = False
                else:
                    df_out_combined = pd.concat([df_out_combined, df_out], axis=1)
                    df_channel_combinations_combined = \
                        pd.concat([df_channel_combinations_combined, df_channel_combinations])

                # write intermediate combination results
                try:
                    df_out_combined = df_out_combined \
                        .reindex(['Signal', 'Global'].extend(SIGNALS_TO_EVALUATE), level=0)
                except:
                    continue

                df_out_combined.transpose().to_csv(
                    os.path.normpath(os.path.join(path, f'Intermediate_combined_results.csv')))
                df_channel_combinations_combined.to_csv(
                    os.path.normpath(os.path.join(path, f'Intermediate_combined_combinations.csv')))

            except (OSError, ErrorForDisplay) as e:
                if dev_run:
                    print(f'Expectable error in file {abs_child}:\n {e}')
                if not dev_run:
                    error_messages = error_messages + f'Error in file {abs_child}:\n {e}\n' \
                                                      f'Full error message can be found in log file.\n\n'

                logging.error(f'Error in file {abs_child}:\n {e}')
                logging.error(traceback.format_exc())
                reading_problems.append(abs_child)
                continue

            except BaseException as e:
                if dev_run:
                    print(f'Unexpected error in file {abs_child}:\n {e}')

                if not dev_run:
                    error_messages = error_messages + f'Unexpected error in file {abs_child}:\n {e}\n' \
                                                      f'Full error message in log file. Please contact developer.\n\n'

                logging.error(f'Unexpected error in file {abs_child}:\n {e}')
                logging.error(traceback.format_exc())
                reading_problems.append(abs_child)
                continue
    try:
        if not df_out_combined.empty:
            df_out_combined = df_out_combined \
                .reindex(['Signal', 'Global'].extend(SIGNALS_TO_EVALUATE), level=0)
        df_out_combined.transpose() \
            .to_excel(os.path.normpath(
            os.path.join(path,
                         f'RBDtector_combined_results_{str(datetime.now()).replace(" ", "_").replace(":", "-")}.xlsx')))
        df_channel_combinations_combined \
            .to_excel(os.path.normpath(
            os.path.join(path,
                         f'Channel_combinations_combined_{str(datetime.now()).replace(" ", "_").replace(":", "-")}.xlsx')))

    except OSError as e:
        if dev_run:
            print(f'Error while writing output summary:\n {e}')
        if not dev_run:
            error_messages = error_messages + f'Error while writing output summary:\n {e}' \
                                              f'Full error message can be found in log file.\n\n'

        logging.error(f'Error in file {abs_child}:\n {e}')
        logging.error(traceback.format_exc())
        reading_problems.append(abs_child)

    if len(reading_problems) != 0:
        logging.error(f'These files could not be processed: {reading_problems}')
        print(f'These files could not be read: {reading_problems}')
        error_messages = f'These files could not be processed: {reading_problems}\n\n' + error_messages

    return error_messages
