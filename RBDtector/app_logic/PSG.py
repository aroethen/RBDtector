# internal modules
import os
from typing import Tuple

from app_logic.dataframe_creation import DataframeCreation
from data_structures.annotation_data import AnnotationData
from data_structures.raw_data import RawData
from input_handling import input_reader as ir
from output import csv_writer
from visualization.dev_plots import dev_plots

# python modules
import logging

# dependencies
import numpy as np
import pandas as pd

# DEFINITIONS
from util.definitions import BASELINE_NAME, EVENT_TYPE, HUMAN_RATING_LABEL, SLEEP_CLASSIFIERS
from util.settings import Settings


class PSG:
    """ Perform automated detection of motoric arousals during REM sleep consistent with RBD.

    :param input_path: absolute path to directory that contains an EDF file to be evaluated
    and all relevant annotation files
    :param output_path: absolute path to directory in which to create the result files
    """
    def __init__(self, input_path: str = '', output_path: str = ''):
        self._input_path = input_path
        self._output_path = output_path

        self._raw_data: RawData = None                  # content of edf file
        self._annotation_data: AnnotationData = None    # content of txt files
        self._calculated_data: pd.DataFrame = None      # dataframe with all currently calculated data

        logging.info('Definitions:\n'
                     f'BASELINE_NAME = {str(BASELINE_NAME)}\n'
                     f'HUMAN_RATING_LABEL = {str(HUMAN_RATING_LABEL)}\n'
                     f'EVENT_TYPE = {str(EVENT_TYPE)}\n'
                     f'{str(Settings.to_string())}'
                     )
        logging.debug('New PSG Object created')

# PUBLIC FUNCTIONS
    @property
    def input_path(self):
        return self._input_path

    @input_path.setter
    def input_path(self, input_path):
        self._input_path = input_path

    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, output_path):
        self._output_path = output_path


### OLD 'monolith' WORKFLOW

    def generate_output(self):
        if not Settings.DEV:
            logging.debug('PSG starting to generate output')

            # read input data
            logging.debug('PSG starting to read input')
            self._raw_data, self._annotation_data = ir.read_input(self._input_path, Settings.SIGNALS_TO_EVALUATE)

            # calculate RBD event scorings
            self._calculated_data = self._process_data()

        if Settings.DEV:
            pickle_path = os.path.join(self._input_path, 'pickledDF')
            self._calculated_data = pd.read_pickle(pickle_path)
            logging.info(f'Used pickled calculation dataframe from {pickle_path}')

        if Settings.SHOW_PLOT:
            dev_plots(self._calculated_data, self.output_path)

        # pickle for further DEV and stats_script use
        pickle_path = os.path.join(self.output_path, 'pickledDF')
        self._calculated_data.to_pickle(pickle_path)
        logging.info(f'Pickled dataframe stored at {pickle_path}')

        csv_writer.write_output(self._output_path,
                                calculated_data=self._calculated_data,
                                signal_names=Settings.SIGNALS_TO_EVALUATE)


    def _process_data(self) -> pd.DataFrame:

        signal_names = Settings.SIGNALS_TO_EVALUATE.copy()

        # extract start of PSG, sample rate of chin EMG channel and number of chin EMG samples to create datetime index
        start_datetime = self._raw_data.get_header()['startdate']
        sample_rate = self._raw_data.get_data_channels()['EMG'].get_sample_rate()
        sample_length = len(self._raw_data.get_data_channels()['EMG'].get_signal())

        # prepare DataFrame with DatetimeIndex
        idx = DataframeCreation.create_datetime_index(start_datetime, sample_rate, sample_length)
        df = pd.DataFrame(index=idx)

        # add sleep profile to df
        df['sleep_phase'], df['is_REM'] = self.create_sleep_profile_column(df.index, self._annotation_data)

        # add artifacts to df
        df['is_artifact'] = self.add_artifacts_to_df(idx, self._annotation_data, Settings.FLOW)

        # cut off all samples before start of sleep_profile assessment if it exists
        start_of_first_full_sleep_phase = df['sleep_phase'].ne(SLEEP_CLASSIFIERS['artifact']).idxmax()
        df = df[start_of_first_full_sleep_phase:]

        # find all epochs and miniepochs of global artifact-free REM sleep
        df['artifact_free_rem_sleep_epoch'], df['artifact_free_rem_sleep_miniepoch'] = \
            self.find_global_artifact_free_REM_sleep_epochs_and_miniepochs(df.index, df['is_artifact'].squeeze(),
                                                                           df['is_REM'].squeeze())

        # process human rating for artifact evaluation per signal and event
        human_rating = self._annotation_data.human_rating[0][1]
        human_rating_label_dict = human_rating.groupby('event').groups
        logging.debug(human_rating_label_dict)

        # TODO: remove human artifact data in final product
        # process second human rating for artifact extraction per signal and event
        human_rating2 = self._annotation_data.human_rating[1][1]
        human_rating_label_dict2 = human_rating2.groupby('event').groups
        logging.debug(human_rating_label_dict2)

        # FOR EACH EMG SIGNAL:
        for signal_type in signal_names.copy():
            logging.debug(signal_type + ' start')

            # Check if signal type exists in edf file
            try:
                signal_array = self._raw_data.get_data_channels()[signal_type].get_signal()
            except KeyError:
                signal_names.remove(signal_type)
                continue

            # Resample to 256 Hz
            sample_rate = self._raw_data.get_data_channels()[signal_type].get_sample_rate()
            df[signal_type] = DataframeCreation.signal_to_hz_rate_datetimeindexed_series(
                Settings.RATE, sample_rate, signal_array, signal_type, start_datetime)[start_of_first_full_sleep_phase:]

            # add signal type baseline column
            df = self.add_signal_baseline_to_df(df, self._annotation_data, signal_type)

            # add human rating boolean arrays
            df = self.add_human_rating_for_signal_type_to_df(df, human_rating, human_rating_label_dict, signal_type)

            # find increased activity
            df[signal_type + '_isGE2xBaseline'] = df[signal_type].abs() >= 2 * df[signal_type + '_baseline']

            df[signal_type + '_two_times_baseline_and_valid'] = \
                df[signal_type + '_isGE2xBaseline'] & df['artifact_free_rem_sleep_miniepoch']
            if Settings.HUMAN_ARTIFACTS:
                df[signal_type + '_two_times_baseline_and_valid'] = \
                    df[signal_type + '_two_times_baseline_and_valid'] & (~df[signal_type + '_human_artifact'])

            df = self.find_increased_activity(df, signal_type, Settings.COUNT_BASED_ACTIVITY)

            # find increased sustained activity
            df = self.find_increased_sustained_activity(df, signal_type)

            # find tonic epochs and adapt baseline if necessary
            df = self.find_tonic_activity_and_adapt_baseline(df, signal_type)

            # repeat search for sustained activity according to new baseline if necessary
            if df[signal_type + '_tonic'].any():
                df[signal_type + '_isGE2xBaseline'] = df[signal_type].abs() >= 2 * df[signal_type + '_baseline']
                df[signal_type + '_two_times_baseline_and_valid'] = \
                    df[signal_type + '_isGE2xBaseline'] & df['artifact_free_rem_sleep_miniepoch']
                if Settings.HUMAN_ARTIFACTS:
                    df[signal_type + '_two_times_baseline_and_valid'] = \
                        df[signal_type + '_two_times_baseline_and_valid'] & (~df[signal_type + '_human_artifact'])
                df = self.find_increased_activity(df, signal_type, Settings.COUNT_BASED_ACTIVITY)
                df = self.find_increased_sustained_activity(df, signal_type)

            # find phasic activity and miniepochs
            df = self.find_phasic_activity_and_miniepochs(df, signal_type)

            df = self.find_any_activity_and_miniepochs(df, signal_type)
            logging.debug(signal_type + ' end')

        # if VERBOSE:
        #     dev_plots(df, self.output_path)
        return df


### NEW GRANULAR WORKFLOW

    def use_pickled_df_as_calculated_data(self, pickle_path: str) -> None:
        """ 'Setter' method for development. Sets the pickled pandas DataFrame at 'pickle_path' as
        'self._calculated_data'.
        :param pickle_path: absolute path of pickle file containing a previously pickled DataFrame with calculated data
        """
        if self._calculated_data is not None:
            logging.info(f'Previous calculations are overwritten by content of pickle file.')

        self._calculated_data = pd.read_pickle(pickle_path)
        logging.info(f'Used pickled calculation dataframe from {pickle_path}')

    def read_input(self, signals_to_evaluate, read_human_rating):
        """
        Read input data from ``input_path`` as specified at PSG object instantiation

        :param signals_to_evaluate: Names of EMG signals as specified in EDF signal header
        :return raw_data, annotation_data: RawData filled with EDF data and AnnotationData filled with the given
        text file data
        """
        logging.debug('PSG starting to read input')

        raw_data, annotation_data = ir.read_input(self._input_path, signals_to_evaluate,
                                                  read_human_rating=read_human_rating)

        logging.debug('PSG finished reading input')

        return raw_data, annotation_data

    @staticmethod
    def prepare_evaluation(raw_data, annotation_data, signal_names, assess_flow_events):

        # extract start of PSG, sample rate of chin EMG channel and number of chin EMG samples to create datetime index
        start_datetime = raw_data.get_header()['startdate']
        sample_rate = raw_data.get_data_channels()['EMG'].get_sample_rate()
        sample_length = len(raw_data.get_data_channels()['EMG'].get_signal())

        # prepare DataFrame with DatetimeIndex
        idx = DataframeCreation.create_datetime_index(start_datetime, sample_rate, sample_length)
        df = pd.DataFrame(index=idx)

        # add sleep profile to df
        df['sleep_phase'], df['is_REM'] = PSG.create_sleep_profile_column(df.index, annotation_data)

        # cut off all samples before start of sleep_profile assessment if it exists
        start_of_first_full_sleep_phase = df['sleep_phase'].ne(SLEEP_CLASSIFIERS['artifact']).idxmax()
        df = df[start_of_first_full_sleep_phase:]

        # add signals to DataFrame
        for signal_type in signal_names.copy():
            logging.debug(signal_type + ' start')

            # Check if signal type exists in edf file
            try:
                signal_array = raw_data.get_data_channels()[signal_type].get_signal()
            except KeyError:
                signal_names.remove(signal_type)
                continue

            # Resample to 256 Hz and add to df
            sample_rate = raw_data.get_data_channels()[signal_type].get_sample_rate()
            df[signal_type] = DataframeCreation.signal_to_hz_rate_datetimeindexed_series(
                Settings.RATE, sample_rate, signal_array, signal_type, start_datetime)[start_of_first_full_sleep_phase:]

        # add global artifacts to df
        df['is_global_artifact'] = PSG.add_artifacts_to_df(idx, annotation_data, assess_flow_events)

        return df[signal_names], df['is_REM'], df['is_global_artifact'], signal_names

    @staticmethod
    def find_global_artifact_free_REM_sleep_epochs_and_miniepochs(idx: pd.DatetimeIndex,
                                                                  artifact_signal_series: pd.Series,
                                                                  is_REM_series: pd.Series):

        df = pd.DataFrame(index=idx)

        artifact_in_3s_miniepoch = artifact_signal_series \
            .resample('3s') \
            .sum() \
            .gt(0)
        df['miniepoch_contains_artifact'] = artifact_in_3s_miniepoch
        df['miniepoch_contains_artifact'] = df['miniepoch_contains_artifact'].ffill()
        df['artifact_free_rem_sleep_miniepoch'] = is_REM_series & ~df['miniepoch_contains_artifact']

        # find all 30s epochs of global artifact-free REM sleep for tonic event detection
        artifact_in_30s_epoch = artifact_signal_series \
            .resample('30s') \
            .sum() \
            .gt(0)
        df['epoch_contains_artifact'] = artifact_in_30s_epoch
        df['epoch_contains_artifact'] = df['epoch_contains_artifact'].ffill()
        df['artifact_free_rem_sleep_epoch'] = is_REM_series & ~df['epoch_contains_artifact']

        return df['artifact_free_rem_sleep_epoch'], df['artifact_free_rem_sleep_miniepoch']

    @staticmethod
    def prepare_human_signal_artifacts(annotation_data, index, signal_names):
        """

        :param annotation_data:
        :param index:
        :param signal_names:
        :return: pd.DataFrame with index 'index' containing 3 columns per signal in 'signal_names'.
        Their column names follow the following schematic:
            - '<signal_name>_human_artifact'  (artifacts of either human rater)
            - '<signal_name>_human1_artifact' (artifacts of human rater 1)
            - '<signal_name>_human2_artifact' (artifacts of human rater 2)
        """

        try:
            # process human rating for artifact evaluation per signal and event
            human_rating1 = annotation_data.human_rating[0][1]
            human_rating_label_dict1 = human_rating1.groupby('event').groups
            logging.debug(human_rating_label_dict1)
        except IndexError as e:
            raise FileNotFoundError("Human rating does not exist.")

        try:
            # process second human rating for artifact extraction per signal and event
            human_rating2 = annotation_data.human_rating[1][1]
            human_rating_label_dict2 = human_rating2.groupby('event').groups
            logging.debug(human_rating_label_dict2)
        except IndexError as e:
            logging.log("Only one human rater file found.")
            human_rating2 = None

        df_artifacts = pd.DataFrame(index=index)

        for signal_type in signal_names.copy():

            #add human rating boolean arrays
            df_artifacts[signal_type + '_human1_artifact'] = \
                PSG.transform_human_artifact_rating_for_signal_type_to_series(
                human_rating1, human_rating_label_dict1, signal_type, df_artifacts.index)

            df_artifacts[signal_type + '_human_artifact'] = df_artifacts[signal_type + '_human1_artifact']

            if human_rating2 is not None:
                df_artifacts[signal_type + '_human2_artifact'] = \
                    PSG.transform_human_artifact_rating_for_signal_type_to_series(
                    human_rating2, human_rating_label_dict2, signal_type, df_artifacts.index)

                # merge artifacts of both raters
                df_artifacts[signal_type + '_human_artifact'] = np.logical_or(
                    df_artifacts[signal_type + '_human1_artifact'],
                    df_artifacts[signal_type + '_human2_artifact']
                )

        return df_artifacts

    @staticmethod
    def find_signal_artifact_free_REM_sleep_epochs_and_miniepochs(index, is_REM_series, is_global_artifact_series,
                                                                  signal_artifacts, signal_names):
        df = pd.DataFrame(index=index)
        df_help = pd.DataFrame(index=index)
        signal_artifacts_used = 1

        # find artifact-free REM sleep miniepochs per signal:
        for signal_name in signal_names:

            # global and signal artifacts
            if signal_artifacts is not None:
                artifact_signal_series = np.logical_or(
                    is_global_artifact_series, signal_artifacts[signal_name + '_signal_artifact'])
            else:
                artifact_signal_series = is_global_artifact_series
                signal_artifacts_used = 0

            # artifact-free miniepochs for signal
            artifact_in_3s_miniepoch = artifact_signal_series \
                .resample('3s') \
                .sum() \
                .gt(0)
            # TODO: Find out whether df_help is necessary or can be skipped
            df_help['miniepoch_contains_artifact'] = artifact_in_3s_miniepoch
            df_help['miniepoch_contains_artifact'] = df_help['miniepoch_contains_artifact'].ffill()
            df[signal_name + '_artifact_free_rem_sleep_miniepoch'] = is_REM_series & ~df_help['miniepoch_contains_artifact']

            # artifact-free epochs for signal
            artifact_in_30s_epoch = artifact_signal_series \
                .resample('30s') \
                .sum() \
                .gt(0)
            df_help['epoch_contains_artifact'] = artifact_in_30s_epoch
            df_help['epoch_contains_artifact'] = df_help['epoch_contains_artifact'].ffill()
            df[signal_name + '_artifact_free_rem_sleep_epoch'] = is_REM_series & ~df_help['epoch_contains_artifact']

            if not signal_artifacts_used:
                logging.info('No special signal artifacts were used. All signals use global artifacts only.')

        return df


    @staticmethod
    def detect_rbd_events(df_signals, artifact_free_rem_sleep_per_signal, signal_names, annotation_data):
        """
        Detects RBD events in

        :return: ``self`` instance of PSG
        """
        df = df_signals.copy()

        # FOR EACH EMG SIGNAL:
        for signal_name in signal_names:

            # add signal type baseline column
            df = PSG.add_signal_baseline_to_df(df, annotation_data, signal_name)

            # find increased activity
            df[signal_name + '_isGE2xBaseline'] = df[signal_name].abs() >= 2 * df[signal_name + '_baseline']
            df[signal_name + '_isGE2xBaseline'] = np.logical_and(
                df[signal_name + '_isGE2xBaseline'],
                artifact_free_rem_sleep_per_signal[signal_name + '_artifact_free_rem_sleep_miniepoch'])

            df = PSG.find_increased_activity(
                df, signal_name,
                artifact_free_rem_sleep_per_signal[signal_name + '_artifact_free_rem_sleep_miniepoch'],
                Settings.COUNT_BASED_ACTIVITY)

            # find increased sustained activity
            df = PSG.find_increased_sustained_activity(df, signal_name)

            # find tonic epochs and adapt baseline if necessary
            df = PSG.find_tonic_activity_and_adapt_baseline(
                df, signal_name, artifact_free_rem_sleep_per_signal[signal_name + '_artifact_free_rem_sleep_epoch'])

            # repeat search for sustained activity according to new baseline if necessary
            if df[signal_name + '_tonic'].any():
                df[signal_name + '_isGE2xBaseline'] = df[signal_name].abs() >= 2 * df[signal_name + '_baseline']
                df[signal_name + '_isGE2xBaseline'] = np.logical_and(
                    df[signal_name + '_isGE2xBaseline'],
                    artifact_free_rem_sleep_per_signal[signal_name + '_artifact_free_rem_sleep_miniepoch'])

                df = PSG.find_increased_activity(
                    df, signal_name,
                    artifact_free_rem_sleep_per_signal[signal_name + '_artifact_free_rem_sleep_miniepoch'],
                    Settings.COUNT_BASED_ACTIVITY)
                df = PSG.find_increased_sustained_activity(df, signal_name)

            # find phasic activity and miniepochs
            df = PSG.find_phasic_activity_and_miniepochs(df, signal_name)
            df[signal_name + '_phasic_miniepochs'] = np.logical_and(
                df[signal_name + '_phasic_miniepochs'],
                artifact_free_rem_sleep_per_signal[signal_name + '_artifact_free_rem_sleep_miniepoch'])

            df = PSG.find_any_activity_and_miniepochs(df, signal_name)
            df[signal_name + '_any_miniepochs'] = np.logical_and(
                df[signal_name + '_any_miniepochs'],
                artifact_free_rem_sleep_per_signal[signal_name + '_artifact_free_rem_sleep_miniepoch'])

        return df

### Modernized internal methods
    @staticmethod
    def transform_human_artifact_rating_for_signal_type_to_series(human_rating, human_rating_label_dict, signal_type,
                                                                  index):
        artifact_label = EVENT_TYPE['artifact']

        df = pd.DataFrame(index=index)

        # Create column for human rating of event type
        df[signal_type + '_human_artifact'] = pd.Series(False, index=df.index)
        logging.debug(HUMAN_RATING_LABEL[signal_type] + artifact_label)

        # Get relevant artifact annotations for column
        human_event_type_indices = \
            human_rating_label_dict.get(HUMAN_RATING_LABEL[signal_type] + artifact_label, [])

        # Set bool column true in all rows with annotated indices
        for idx in human_event_type_indices:
            df.loc[
                human_rating.iloc[idx]['event_onset']:human_rating.iloc[idx]['event_end_time'],
                [signal_type + '_human_artifact']
            ] = True

        return df[signal_type + '_human_artifact']

    @staticmethod
    def add_artifacts_to_df(idx: pd.DatetimeIndex, annotation_data: AnnotationData, assess_flow_events: bool):
        arousals: pd.DataFrame = annotation_data.arousals[1]

        df = pd.DataFrame(index=idx)
        df['artifact_event'] = pd.Series(False, index=df.index)
        for label, on, off in zip(arousals['event'], arousals['event_onset'], arousals['event_end_time']):
            df.loc[on:off, ['artifact_event']] = True

        if assess_flow_events:
            flow_events = annotation_data.flow_events[1]
            df['flow_event'] = pd.Series(False, index=df.index)
            for label, on, off in zip(flow_events['event'], flow_events['event_onset'], flow_events['event_end_time']):
                df.loc[on:off, ['flow_event']] = True

        # add conditional column 'is_artifact'
        if assess_flow_events:
            df['is_artifact'] = np.logical_or(df['artifact_event'], df['flow_event'])
        else:
            df['is_artifact'] = df['artifact_event']

        return df['is_artifact']

    @staticmethod
    def create_sleep_profile_column(idx: pd.DatetimeIndex, annotation_data: AnnotationData) \
            -> Tuple[pd.Series, pd.Series]:
        """
        :param index: DatetimeIndex to be used for sleep profile column
        :param sleep_profile: Pandas DataFrame with categorical column "sleep_phase" containing sleeping phase
                classification strings. index is a DatetimeIndex containing the timestamps from the sleep profile input
                text file
        :returns df['sleep_phase'], df['is_REM']: 2 pd.Series sleep_phase (categorical) and is_REM (bool) respectively,
                with index idx
        """
        df = pd.DataFrame(index=idx)
        sleep_profile = annotation_data.sleep_profile[1]

        sleep_profile.sort_index(inplace=True)

        # if first time stamp of sleep profile is before first timestamp of emg data, set first sleep profile entry to
        # artifact
        if sleep_profile.index.min() < df.index[0]:
            sleep_profile.loc[sleep_profile.index.min(), 'sleep_phase'] = SLEEP_CLASSIFIERS['artifact']

        # append a final row to sleep profile with "Artifact" sleep phase
        # for easier concatenation with df while correctly accounting for last 30s sleep profile interval
        sleep_profile = sleep_profile.append(pd.DataFrame({'sleep_phase': SLEEP_CLASSIFIERS['artifact']},
                                                          index=[sleep_profile.index.max() + pd.Timedelta('30s')])
                                             )

        sleep_profile.sort_index(inplace=True)

        # resample sleep profile from 2Hz(30s intervals) to Settings.RATE Hz, fill all entries with the correct sleeping phase
        # and add it as column to dataframe
        resampled_sleep_profile = sleep_profile.resample(str(1000 / Settings.RATE) + 'ms').ffill()
        df = pd.concat([df, resampled_sleep_profile], axis=1, join='inner')
        df['is_REM'] = df['sleep_phase'] == SLEEP_CLASSIFIERS['REM']
        return df['sleep_phase'], df['is_REM']


### Old df- and column-dependent internal methods
    @staticmethod
    def find_any_activity_and_miniepochs(df, signal_type):
        """
        Find any activity and 3s miniepochs containing any activity
        :param df: pandas.Dataframe containing columns
                    df[signal_type + '_sustained_activity'] and df[signal_type + '_tonic']
        :param signal_type: name of signal as str as used in the naming of the relevant df columns
        :return: original df with added columns df[signal_type + '_any'] and
                df[signal_type + '_any_miniepochs']
        """

        # find 'any' activity
        df[signal_type + '_any'] = df[signal_type + '_sustained_activity'] | df[signal_type + '_tonic']

        # find 'any' miniepochs
        any_in_3s_miniepoch = df[signal_type + '_any'].squeeze() \
            .resample('3s') \
            .sum() \
            .gt(0)
        df[signal_type + '_any_miniepochs'] = any_in_3s_miniepoch
        df[signal_type + '_any_miniepochs'] = df[signal_type + '_any_miniepochs'].ffill()

        return df

    @staticmethod
    def find_phasic_activity_and_miniepochs(df, signal_type):
        """
        Find phasic activity and 3s miniepochs containing phasic activity
        :param df: pandas.Dataframe containing column df[signal_type + '_sustained_activity']
        :param signal_type: name of signal as str as used in the naming of the df[signal_type + '_sustained_activity']
                column
        :return: original df with added columns df[signal_type + '_phasic'] and
                df[signal_type + '_phasic_miniepochs']
        """

        # find phasic activity
        continuous_phasic_sustained_activity = df[signal_type + '_sustained_activity'] \
            .groupby([df[signal_type + '_sustained_activity'].diff().ne(0).cumsum()]) \
            .transform('size') \
            .le(Settings.RATE * 5)
        df[signal_type + '_phasic'] = continuous_phasic_sustained_activity & df[signal_type + '_sustained_activity']

        # find phasic miniepochs
        phasic_in_3s_miniepoch = df[signal_type + '_phasic'].squeeze() \
            .resample('3s') \
            .sum() \
            .gt(0)
        df[signal_type + '_phasic_miniepochs'] = phasic_in_3s_miniepoch
        df[signal_type + '_phasic_miniepochs'] = df[signal_type + '_phasic_miniepochs'].ffill()

        return df

    @staticmethod
    def find_tonic_activity_and_adapt_baseline(df, signal_type, artifact_free_rem_sleep_epochs):
        """
        Find tonic activity among 30s intervals of sustained activity signal. Adapt baseline to RMS of tonic activity
        during exact time of sustained activity inside tonic epoch.

        :param df: pandas.Dataframe containing column df[signal_type + '_sustained_activity']
        :param signal_type: name of signal as str as used in the naming of the df[signal_type + '_sustained_activity']
                column
        :param artifact_free_rem_sleep_epochs: Series of artifact-free REM sleep epochs for this signal
        :return: original dataframe with column df[signal_type + '_tonic'] added and
                column df[signal_type + '_baseline'] updated to tonic activity
        """

        sustained_signal = np.logical_and(df[signal_type + '_sustained_activity'], artifact_free_rem_sleep_epochs)
        tonic_in_30s = sustained_signal \
            .resample('30s') \
            .sum() \
            .gt((Settings.RATE * 30) / 2)
        df[signal_type + '_tonic'] = tonic_in_30s
        df[signal_type + '_tonic'] = df[signal_type + '_tonic'].ffill()
        df[signal_type + '_tonic'] = np.logical_and(df[signal_type + '_tonic'], artifact_free_rem_sleep_epochs)

        activity_at_tonic_interval = df[signal_type + '_tonic'] & df[signal_type + '_sustained_activity']
        rms_per_tonic = df.loc[activity_at_tonic_interval, signal_type]
        rms_per_tonic = rms_per_tonic ** 2
        rms_per_tonic = rms_per_tonic.groupby([activity_at_tonic_interval.diff().ne(0).cumsum()]) \
            .transform(lambda x: np.sqrt(np.mean(x)))
        df.loc[activity_at_tonic_interval, [signal_type + '_baseline']] = rms_per_tonic[activity_at_tonic_interval]

        return df

    @staticmethod
    def find_increased_sustained_activity(df, signal_type):
        """
        Finds increased sustained activity defined as increased activity of at least MIN_SUSTAINED seconds with no gaps bigger
        than MAX_GAP_SIZE.
        :param df: pandas.Dataframe that contains a datetimeindexed column df[signal_type + '_increased_activity']
        :param signal_type: name of signal as str as used in the naming of the df[signal_type + '_increased_activity']
                column
        :return: original df with added column df[signal_type + '_sustained_activity']
        """
        continuous_vals_gt_min_start = df[signal_type + '_increased_activity'] \
            .groupby([df[signal_type + '_increased_activity'].diff().ne(0).cumsum()]) \
            .transform('size') \
            .gt(Settings.RATE * Settings.MIN_SUSTAINED)
        min_sustained_activity = \
            continuous_vals_gt_min_start & df[signal_type + '_increased_activity']

        continuous_vals_gt_min_end = df[signal_type + '_increased_activity'] \
            .groupby([df[signal_type + '_increased_activity'].diff().ne(0).cumsum()]) \
            .transform('size') \
            .lt(Settings.RATE * Settings.MAX_GAP_SIZE)
        max_tolerable_gaps = \
            continuous_vals_gt_min_end & ~df[signal_type + '_increased_activity']

        # iteratively add gaps and smaller activity windows to sustained activity, as long as they fit the sustained
        # activity criteria
        sustained_activity = pd.Series()
        new_sustained_activity = min_sustained_activity

        while not sustained_activity.equals(new_sustained_activity):
            sustained_activity = new_sustained_activity

            # add gaps < MAX_GAP_SIZE to sustained activity periods if they lie directly to the right of a
            # sustained activity interval
            diff_signal = \
                new_sustained_activity.astype(int) \
                + max_tolerable_gaps.astype(int) * 3
            relevant_diffs = diff_signal.diff().replace(to_replace=0, method='ffill').eq(2).fillna(0).astype(bool)
            new_sustained_activity = new_sustained_activity | relevant_diffs

            # add gaps < 0.25s to sustained activity periods if they lie directly to the left of a new
            # sustained activity interval
            diff_signal = \
                new_sustained_activity.astype(int) \
                + max_tolerable_gaps.astype(int) * 3
            relevant_diffs = diff_signal.diff(-1).replace(to_replace=0, method='bfill').eq(2).fillna(0).astype(bool)
            new_sustained_activity = new_sustained_activity | relevant_diffs

            # add activity to sustained activity periods if they lie directly to the right of a new
            # sustained activity interval
            non_sustained_activity = df[signal_type + '_increased_activity'] & ~new_sustained_activity
            diff_signal = \
                new_sustained_activity.astype(int) \
                + non_sustained_activity.astype(int) * 3
            relevant_diffs = diff_signal.diff().replace(to_replace=0, method='ffill').eq(2).fillna(0).astype(bool)
            new_sustained_activity = new_sustained_activity | relevant_diffs

            # add activity to sustained activity periods if they lie directly to the left of a new
            # sustained activity interval
            non_sustained_activity = df[signal_type + '_increased_activity'] & ~new_sustained_activity
            diff_signal = \
                new_sustained_activity.astype(int) \
                + non_sustained_activity.astype(int) * 3
            relevant_diffs = diff_signal.diff(-1).replace(to_replace=0, method='bfill').eq(2).fillna(0).astype(bool)
            new_sustained_activity = new_sustained_activity | relevant_diffs

            # df[signal_type + '_new_sustained_activity'] = new_sustained_activity
            # df['diffs'] = relevant_diffs
        df[signal_type + '_sustained_activity'] = sustained_activity
        return df

    @staticmethod
    def find_increased_activity(df: pd.DataFrame, signal_type: str, artifact_free_rem_sleep_miniepoch,
                                count_based_activity: bool = False):
        """

        :param df: Datetimeindexed pandas.Dataframe containing columns:
            df[signal_type]: float
            df[signal_type + '_baseline']
        :param signal_type: Signal type to find and create df[signal_type + '_increased_activity'] for
        :param artifact_free_rem_sleep_miniepoch: boolean series of artifact free miniepochs of this signal
        :param count_based_activity:
            True: increased activity defined as more than 3 values of a 50ms interval bigger than 2x baseline
            False: increased activity defined as RMS of a 50ms interval is bigger than 2x baseline
        :return: Updated Dataframe
        """

        if count_based_activity:
            series_to_resample = df[signal_type].astype(int)
            increased_in_point05s = series_to_resample \
                .resample(Settings.CHUNK_SIZE) \
                .sum() \
                .apply(lambda x: x > 3)
            increased_in_point05s = increased_in_point05s.resample(Settings.FREQ).ffill()
            df[signal_type + '_increased_activity'] = increased_in_point05s
            df[signal_type + '_increased_activity'] = df[signal_type + '_increased_activity']

            if Settings.WITH_OFFSET:
                increased_in_point05s_with_offset = series_to_resample \
                    .resample(Settings.CHUNK_SIZE, offset=Settings.OFFSET_SIZE) \
                    .sum() \
                    .apply(lambda x: x > 3)
                increased_in_point05s_with_offset = increased_in_point05s_with_offset.resample(Settings.FREQ).ffill()
                df[signal_type + '_increased_activity_with_offset'] = increased_in_point05s_with_offset
                df[signal_type + '_increased_activity'] = np.logical_or(
                    df[signal_type + '_increased_activity'],
                    df[signal_type + '_increased_activity_with_offset']
                ).fillna(False)

        else:
            valid_signal = artifact_free_rem_sleep_miniepoch * df[signal_type]
            valid_signal = valid_signal.pow(2)
            increased_in_point05s = valid_signal \
                .resample(Settings.CHUNK_SIZE) \
                .mean() \
                .apply(np.sqrt)

            df[signal_type + '_increased_activity'] = increased_in_point05s.resample(Settings.FREQ).ffill()
            df[signal_type + '_increased_activity'] = df[signal_type + '_increased_activity'].ffill()
            df[signal_type + '_increased_activity'] = \
                df[signal_type + '_increased_activity'] > (2 * df[signal_type + '_baseline'])

            if Settings.WITH_OFFSET:
                valid_signal = artifact_free_rem_sleep_miniepoch * df[signal_type]
                valid_signal = valid_signal.pow(2)
                increased_in_point05s_with_offset = valid_signal \
                    .resample(Settings.CHUNK_SIZE, offset=Settings.OFFSET_SIZE) \
                    .mean() \
                    .apply(np.sqrt)
                increased_in_point05s_with_offset = increased_in_point05s_with_offset.resample(Settings.FREQ).ffill()
                increased_activity_with_offset = pd.Series(increased_in_point05s_with_offset, index=df.index)
                increased_activity_with_offset = \
                    increased_activity_with_offset > (2 * df[signal_type + '_baseline'])
                df[signal_type + '_increased_activity'] = np.logical_or(
                    df[signal_type + '_increased_activity'],
                    increased_activity_with_offset
                )
        return df

    def add_human_rating_for_signal_type_to_df(self, df, human_rating, human_rating_label_dict, signal_type):

        # For all event types (tonic, intermediate, phasic, artifact)
        for event_type in EVENT_TYPE.keys():

            # Create column for human rating of event type
            df[signal_type + '_human_' + event_type] = pd.Series(False, index=df.index)
            logging.debug(HUMAN_RATING_LABEL[signal_type] + EVENT_TYPE[event_type])

            # Get relevant annotations for column
            human_event_type_indices = \
                human_rating_label_dict.get(HUMAN_RATING_LABEL[signal_type] + EVENT_TYPE[event_type], [])

            # Set bool column true in all rows with annotated indices
            for idx in human_event_type_indices:
                df.loc[
                    human_rating.iloc[idx]['event_onset']:human_rating.iloc[idx]['event_end_time'],
                    [signal_type + '_human_' + event_type]
                ] = True

        # Add second human rating artifacts to dataframe if available
        if len(self._annotation_data.human_rating) > 1:

            # create rater1-only and rater2-only artifact column
            df[signal_type + '_human1_artifact'] = df[signal_type + '_human_artifact']
            df[signal_type + '_human2_artifact'] = pd.Series(False, index=df.index)

            second_rater = self._annotation_data.human_rating[1][1]
            human_rating_label_dict = second_rater.groupby('event').groups

            # Get relevant annotations for column
            second_rater_event_type_indices = \
                human_rating_label_dict.get(HUMAN_RATING_LABEL[signal_type] + EVENT_TYPE['artifact'], [])

            # Set bool column true in all rows with annotated indices
            for idx in second_rater_event_type_indices:
                df.loc[
                    second_rater.iloc[idx]['event_onset']:second_rater.iloc[idx]['event_end_time'],
                    [signal_type + '_human2_artifact']
                ] = True

            # merge artifacts of both raters
            df[signal_type + '_human_artifact'] = np.logical_or(
                df[signal_type + '_human_artifact'],
                df[signal_type + '_human2_artifact']
            )

        # adaptions to phasic
        df[signal_type + '_human_phasic'] = df[signal_type + '_human_phasic'] & df['artifact_free_rem_sleep_miniepoch']

        phasic_in_3s_miniepoch = df[signal_type + '_human_phasic'].squeeze() \
            .resample('3s') \
            .sum() \
            .gt(0)
        df[signal_type + '_human_phasic_miniepochs'] = phasic_in_3s_miniepoch
        df[signal_type + '_human_phasic_miniepochs'] = df[signal_type + '_human_phasic_miniepochs'].ffill()

        # adaptions to any
        df[signal_type + '_human_any'] = df[signal_type + '_human_tonic'] | \
                                         df[signal_type + '_human_intermediate'] | \
                                         df[signal_type + '_human_phasic']

        any_in_3s_miniepoch = df[signal_type + '_human_any'].squeeze() \
            .resample('3s') \
            .sum() \
            .gt(0)
        df[signal_type + '_human_any_miniepochs'] = any_in_3s_miniepoch
        df[signal_type + '_human_any_miniepochs'] = df[signal_type + '_human_any_miniepochs'].ffill()

        return df

    @staticmethod
    def add_signal_baseline_to_df(df, annotation_data, signal_type):
        bl_start, bl_stop = annotation_data.baseline[BASELINE_NAME[signal_type]]
        baseline = rms(df.loc[bl_start: bl_stop, signal_type])
        df[signal_type + '_baseline'] = baseline
        return df




def rms(np_array_like):
    return np.sqrt(np.mean(np.power(np_array_like.astype(np.double), 2)))

