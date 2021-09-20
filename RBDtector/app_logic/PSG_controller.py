import logging
import os
import pandas as pd

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
            raw_data, annotation_data = psg.read_input(Settings.SIGNALS_TO_EVALUATE,
                                                       read_human_rating=Settings.HUMAN_ARTIFACTS)

            df_signals, is_REM_series, is_global_artifact_series, signal_names = \
                psg.prepare_evaluation(raw_data, annotation_data, Settings.SIGNALS_TO_EVALUATE, Settings.FLOW)

            # find all (mini)epochs of global artifact-free REM sleep
            is_global_artifact_free_rem_sleep_epoch_series, is_global_artifact_free_rem_sleep_miniepoch_series = \
                psg.find_global_artifact_free_REM_sleep_epochs_and_miniepochs(
                    idx=df_signals.index, artifact_signal_series=is_global_artifact_series, is_REM_series=is_REM_series)

            human_signal_artifacts = None
            detected_signal_artifacts = None
            signal_artifacts = None

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

            artifact_free_rem_sleep_per_signal = psg.find_signal_artifact_free_REM_sleep_epochs_and_miniepochs(
                df_signals.index, is_REM_series, is_global_artifact_series, signal_artifacts,
                signal_names)

            rbd_events = psg.detect_rbd_events(df_signals, artifact_free_rem_sleep_per_signal,
                                               signal_names, annotation_data)

            csv_writer.write_output(output_path,
                                    calculated_data=rbd_events,
                                    signal_names=signal_names)


