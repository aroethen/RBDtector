import logging
import os
import pandas as pd

from util.definitions import SLEEP_CLASSIFIERS
from util.settings import Settings
from app_logic.PSG import PSG
from output import csv_writer


class PSGController:
    """High-level controller for PSG evaluation functionality. Provides easy to use functions for RBD event detection."""

    @staticmethod
    def run_rbd_detection(input_path: str, output_path: str):
        psg = PSG(input_path, output_path)

        if Settings.DEV_READ_PICKLE_INSTEAD_OF_EDF:
            psg.use_pickled_df_as_calculated_data(os.path.join(input_path, 'pickledDF'))
        else:
            raw_data, annotation_data = psg.read_input(Settings.SIGNALS_TO_EVALUATE.copy(),
                                                       read_human_rating=Settings.HUMAN_ARTIFACTS)

            df_signals, is_REM_series, is_global_artifact_series, signal_names, sleep_phase_series = \
                psg.prepare_evaluation(raw_data, annotation_data, Settings.SIGNALS_TO_EVALUATE.copy(), Settings.FLOW)

            # find all (mini)epochs of global artifact-free REM sleep
            is_global_artifact_free_rem_sleep_epoch_series, is_global_artifact_free_rem_sleep_miniepoch_series = \
                psg.find_artifact_free_REM_sleep_epochs_and_miniepochs(
                    idx=df_signals.index, artifact_signal_series=is_global_artifact_series, is_REM_series=is_REM_series)


            # FIND SIGNAL ARTIFACTS
            signal_artifacts = pd.DataFrame(index=df_signals.index)
            for signal_name in signal_names:
                signal_artifacts[signal_name + '_signal_artifact'] = 0

            if Settings.HUMAN_ARTIFACTS:
                human_signal_artifacts = psg.prepare_human_signal_artifacts(
                     annotation_data, df_signals.index, signal_names)
                signal_artifacts = pd.DataFrame(index=df_signals.index)
                for signal_name in signal_names:
                    signal_artifacts[signal_name + '_signal_artifact'] = \
                        human_signal_artifacts[signal_name + '_human_artifact']

            if Settings.FIND_ARTIFACTS:
                # TODO implement: detected_signal_artifacts = data.detect_signal_artifacts()
                raise NotImplementedError

            if Settings.SNORE:
                if 'EMG' in signal_names:
                    snore_series = sleep_phase_series.str.lower() == SLEEP_CLASSIFIERS['SNORE'].lower()
                    signal_artifacts['EMG_signal_artifact'] = signal_artifacts['EMG_signal_artifact'] | snore_series

            artifact_free_rem_sleep_per_signal = psg.find_signal_artifact_free_REM_sleep_epochs_and_miniepochs(
                df_signals.index, is_REM_series, is_global_artifact_series, signal_artifacts,
                signal_names)

            # FIND BASELINE
            if Settings.HUMAN_BASELINE:
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
