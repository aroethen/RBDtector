# internal modules
import os

from data_structures.annotation_data import AnnotationData
from data_structures.raw_data import RawData
from input_handling import input_reader as ir
from output import csv_writer

# python modules
import logging

# dependencies
import numpy as np
import pandas as pd
from scipy import interpolate

# dev dependencies
import matplotlib.pyplot as plt
# import os.path

# DEFINITIONS
SPLINES = True
RATE = 256
FREQ = '3.90625ms'
FLOW = True
COUNT_BASED_ACTIVITY = False
DEV = True

BASELINE_NAME = {
    'EMG': 'EMG REM baseline',
    'PLM l': 'PLMl REM baseline',
    'PLM r': 'PLMr REM baseline',
    'AUX': 'AUX REM baseline',
    'Akti.': 'AKTI REM baseline'
}

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

SIGNALS_TO_EVALUATE = ['EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.']


class PSGData:

    def __init__(self, input_path: str = '', output_path: str = ''):
        self._input_path = input_path
        self._output_path = output_path

        self._raw_data: RawData                  # content of edf file
        self._annotation_data: AnnotationData    # content of txt files
        self._calculated_data: pd.DataFrame = pd.DataFrame()
        logging.debug('New PSGData Object created')

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

    def generate_output(self):
        if not DEV:
            logging.debug('PSGData starting to generate output')
            self._read_data()
        self._calculated_data = self._process_data()
        if not DEV:
            human_rater_data = self._annotation_data.get_human_rating()
            csv_writer.write_output(self._output_path, human_rating=human_rater_data)

# PRIVATE FUNCTIONS
    def _read_data(self):
        logging.debug('PSGData starting to read input')
        data = ir.read_input(self._input_path, SIGNALS_TO_EVALUATE)
        self._raw_data = data[0]
        self._annotation_data = data[1]

    def _process_data(self) -> pd.DataFrame:
        # TODO: returns calculated data in form that can be used by output writer
        if not DEV:
            start_datetime = self._raw_data.get_header()['startdate']
            signal_names = SIGNALS_TO_EVALUATE.copy()

            # prepare DataFrame with DatetimeIndex
            idx = self._create_datetime_index(start_datetime)
            df = pd.DataFrame(index=idx)

            # add sleep profile to df
            df = self.add_sleep_profile_to_df(df)

            # add artefacts to df
            df = self.add_artefacts_to_df(df)

            # cut off all samples before start of sleep_profile assessment
            start_of_first_full_sleep_phase = df['sleep_phase'].ne('A').idxmax()
            df = df[start_of_first_full_sleep_phase:]

            # find all phasic miniepochs of artifact-free REM sleep
            artefacts_array = df['is_artefact'].to_numpy()
            artefacts_series = pd.Series(artefacts_array)
            artefacts_in_miniepochs = artefacts_series.groupby(artefacts_series.index // (RATE * 3)).any().repeat(256*3)
            df['miniepoch_contains_artefact'] = artefacts_in_miniepochs.to_numpy()
            df['artefact_free_rem_sleep_miniepoch'] = df['is_REM'] & ~df['miniepoch_contains_artefact']

            # process human rating for evaluation per signal and event
            human_rating = self._annotation_data.human_rating[1]
            human_rating_label_dict = human_rating.groupby('event').groups
            logging.debug(human_rating_label_dict)

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
                df[signal_type] = self._signal_to_256hz_datetimeindexed_series(
                    signal_array, signal_type, start_datetime
                )[start_of_first_full_sleep_phase:]

                # add signal type baseline column
                df = self.add_signal_baseline_to_df(df, signal_type)

                # add human rating boolean arrays
                df = self.add_human_rating_for_signal_type_to_df(df, human_rating, human_rating_label_dict, signal_type)

            df.to_pickle(os.path.join(self.output_path, 'pickledDF'))

        else:
            df = pd.read_pickle(os.path.join(self.output_path, 'pickledDF'))

        for signal_type in ['EMG']:
        # for signal_type in signal_names:      TODO: NACH DEV WIEDER DIESE VARIANTE VERWENDEN!!!
            # TONIC ################################################################################################

            # find all samples with at least 2x baseline in REM
            df[signal_type + '_isGE2xBaseline'] = df[signal_type].abs() >= 2 * df[signal_type + '_baseline']


            df[signal_type + '_two_times_baseline_and_valid'] = \
                df[signal_type + '_isGE2xBaseline'] & df['artefact_free_rem_sleep_miniepoch']

            df = self.find_increased_activity(df, signal_type, COUNT_BASED_ACTIVITY)

        # find increased sustained activity
            # activity_array = df[signal_type + '_increased_activity'].to_numpy()
            # activity_series = pd.Series(activity_array)
            # # increased_series = increased_series.pow(2)
            #
            # point1s_grouper = activity_series.groupby(increased_series.index // int(RATE * 0.1))
            #
            # sustained_before_filling_gaps = point1s_grouper\
            #     .any()\
            #     .repeat(int(RATE * 0.1))
            # df[signal_type + '_sustained_before_filling_gaps'] = \
            #     sustained_before_filling_gaps[:df.index.size].to_numpy()


            # df[signal_type + '_point1continuous'] = df[signal_type + '_point05s_contains_activity']\
            #     .rolling('100ms')\
            #     .apply(lambda x: x.any())\
            #     .asfreq('3.90625ms')\
            #     .bfill()


        logging.debug(signal_type + ' end')

        ############################################################################################################
        # DEV OUTPUT

        print(df.info())
        df = df.iloc[(df.index.size//2)+(df.index.size//8):-(df.index.size//6)]

        fig, ax = plt.subplots()
        # REM PHASES
        # ax.fill_between(df.index.values, df['is_REM']*(-1000), df['is_REM']*1000,
        #                  facecolor='gainsboro', label="is_REM", alpha=0.7)
        ax.fill_between(df.index.values, df['artefact_free_rem_sleep_miniepoch'] * (-750),
                        df['artefact_free_rem_sleep_miniepoch'] * 750,
                        facecolor='#e1ebe8', label="Artefact-free REM sleep miniepoch", alpha=0.7)

        # ACTIVITIES
        ax.fill_between(df.index.values, df['EMG' + '_increased_activity']*(-50),
                        df['EMG' + '_increased_activity']*50, alpha=0.7, facecolor='yellow',
                        label="0.05s contains activity", zorder=4)
        # ax.fill_between(df.index.values, df['EMG' + '_sustained_before_filling_gaps']*(-25),
        #                 df['EMG' + '_sustained_before_filling_gaps']*25, alpha=0.7, facecolor='orange',
        #                 label="sustained_before_filling_gaps", zorder=4)


        # HUMAN RATING OF CHIN EMG
        # ax.fill_between(df.index.values, df['EMG_human_tonic']*(-1000), df['EMG_human_tonic']*1000,
        #                  facecolor='mediumorchid', label="EMG_human_tonic")
        # ax.fill_between(df.index.values, df['EMG_human_intermediate']*(-1000), df['EMG_human_intermediate']*1000,
        #                  facecolor='violet', label="EMG_human_intermediate")
        ax.fill_between(df.index.values, df['EMG_human_phasic']*(-1000), df['EMG_human_phasic']*1000,
                         facecolor='royalblue', label="EMG_human_phasic", alpha=0.7)
        # ax.fill_between(df.index.values, df['EMG_human_artefact']*(-1000), df['EMG_human_artefact']*1000,
        #                  facecolor='maroon', label="EMG_human_artefact")
        # ax.fill_between(df.index.values, df['miniepoch_contains_artefact'] * (-75),
        #                  df['miniepoch_contains_artefact'] * 75,
        #                  facecolor='#993404', label="miniepoch_contains_artefact", alpha=0.7, zorder=4)



        # EMG CHIN
        ax.plot(df.index.values, df['EMG'], c='#313133', label="EMG", zorder=4)
        ax.plot(df['EMG_baseline'], c='mediumseagreen', label="EMG_baseline", zorder=4)
        ax.plot(df['EMG_baseline']*(-1), c='mediumseagreen', zorder=4)
        ax.plot([df.index.values[0], df.index.values[-1]], [0, 0], c='dimgrey')
        ax.scatter(df.index.values, df['EMG'].where(df['EMG' + '_two_times_baseline_and_valid']), s=4, c='lime',
                    label='Increased activity', zorder=4)

        # ARTEFACTS CHIN
        # ax.fill_between(df.index.values, df['is_artefact'] * (-200), df['is_artefact'] * 200,
        #                  facecolor='#dfc27d', label="is_artefact", alpha=0.7, zorder=3)
        ax.legend(loc='upper left', facecolor='white', framealpha=1)
        plt.show()

        # plt.plot(resampled_signal_dtseries.index.values, resampled_signal_dtseries.values)
        # plt.show()
        ############################################################################################################

        return df

    def find_increased_activity(self, df: pd.DataFrame, signal_type: str, count_based_activity: bool = False):
        """
        :param df: Datetimeindexed pandas.Dataframe containing columns:
            df[signal_type]: float
            df['artefact_free_rem_sleep_miniepoch']: bool
            df[signal_type + '_two_times_baseline_and_valid']: bool
            df[signal_type + '_baseline']
        :param signal_type: Signal type to find and create df[signal_type + '_increased_activity'] for
        :param count_based_activity:
            True: increased activity defined as more than 2 values of a 50ms interval bigger than 2x baseline
            False: increased activity defined as RMS of a 50ms interval is bigger than 2x baseline
        :return: Updated Dataframe
        """

        if count_based_activity:
            series_to_resample = df[signal_type + '_two_times_baseline_and_valid'].astype(int)
            increased_in_point05s = series_to_resample \
                .resample('50ms') \
                .sum() \
                .apply(lambda x: x > 2)
            df[signal_type + '_increased_activity'] = increased_in_point05s
            df[signal_type + '_increased_activity'] = df[signal_type + '_increased_activity'].fillna(method='ffill')

            increased_in_point05s = series_to_resample \
                .resample('50ms', offset='25ms') \
                .sum() \
                .apply(lambda x: x > 2)
            df[signal_type + '_increased_activity_with_offset'] = increased_in_point05s.resample(FREQ).ffill()
            df[signal_type + '_increased_activity'] = np.logical_or(
                df[signal_type + '_increased_activity'],
                df[signal_type + '_increased_activity_with_offset']
            )

        else:
            valid_signal = df['artefact_free_rem_sleep_miniepoch'] * df[signal_type]
            valid_signal = valid_signal.pow(2)
            increased_in_point05s = valid_signal \
                .resample('50ms') \
                .mean() \
                .apply(np.sqrt)
            df[signal_type + '_increased_activity'] = increased_in_point05s
            df[signal_type + '_increased_activity'] = df[signal_type + '_increased_activity'].ffill()
            df[signal_type + '_increased_activity'] = \
                df[signal_type + '_increased_activity'] > (2 * df[signal_type + '_baseline'])

            increased_in_point05s_with_offset = valid_signal \
                .resample('50ms', offset='25ms') \
                .mean() \
                .apply(np.sqrt)
            increased_in_point05s_with_offset = increased_in_point05s_with_offset.resample(FREQ).ffill()
            df[signal_type + '_increased_activity_with_offset'] = increased_in_point05s_with_offset
            df[signal_type + '_increased_activity_with_offset'] = \
                df[signal_type + '_increased_activity_with_offset'] > (2 * df[signal_type + '_baseline'])
            df[signal_type + '_increased_activity'] = np.logical_or(
                df[signal_type + '_increased_activity'],
                df[signal_type + '_increased_activity_with_offset']
            )
        return df

    def add_human_rating_for_signal_type_to_df(self, df, human_rating, human_rating_label_dict, signal_type):

        # For all event types (tonic, intermediate, phasic, artefact)
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

        df[signal_type + '_human_any'] = df[signal_type + '_human_tonic'] | \
                                         df[signal_type + '_human_intermediate'] | \
                                         df[signal_type + '_human_phasic']

        return df

    def add_signal_baseline_to_df(self, df, signal_type):
        bl_start, bl_stop = self._annotation_data.baseline[BASELINE_NAME[signal_type]]
        baseline = rms(df.loc[bl_start: bl_stop, signal_type])
        df[signal_type + '_baseline'] = baseline
        return df

    def add_artefacts_to_df(self, df):

        arousals: pd.DataFrame = self._annotation_data.arousals[1]
        df['artefact_event'] = pd.Series(False, index=df.index)
        for label, on, off in zip(arousals['event'], arousals['event_onset'], arousals['event_end_time']):
            df.loc[on:off, ['artefact_event']] = True

        if FLOW:
            flow_events = self._annotation_data.flow_events[1]
            df['flow_event'] = pd.Series(False, index=df.index)
            for label, on, off in zip(flow_events['event'], flow_events['event_onset'], flow_events['event_end_time']):
                df.loc[on:off, ['flow_event']] = True

        # add conditional column 'is_artefact'
        if FLOW:
            df['is_artefact'] = np.logical_or(df['artefact_event'], df['flow_event'])
        else:
            df['is_artefact'] = df['artefact_event']

        return df

    def add_sleep_profile_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        sleep_profile: pd.DataFrame = self._annotation_data.sleep_profile[1]
        # print(sleep_profile.index.max() + pd.Timedelta('30s'))

        # append a final row to sleep profile with "Artifact" sleep phase
        # for easier concatenation with df while correctly accounting for last 30s sleep profile interval
        sleep_profile = sleep_profile.append(pd.DataFrame({'sleep_phase': 'A'},
                                                          index=[sleep_profile.index.max() + pd.Timedelta('30s')])
                                             )
        sleep_profile.sort_index(inplace=True)

        # resample sleep profile from 2Hz(30s intervals) to 256 Hz, fill all entries with the correct sleeping phase
        # and add it as column to dataframe
        resampled_sleep_profile = sleep_profile.resample(str(1000 / RATE) + 'ms').ffill()
        df = pd.concat([df, resampled_sleep_profile], axis=1, join='inner')
        df['is_REM'] = df['sleep_phase'] == "REM"
        return df

    def _create_datetime_index(self, start_datetime):
        freq_in_ms = 1000 / RATE
        emg_channel = self._raw_data.get_data_channels()['EMG']
        sample_rate = emg_channel.get_sample_rate()
        sample_length = len(emg_channel.get_signal())
        index_length = (sample_length / sample_rate) * RATE
        return pd.date_range(start_datetime, freq=str(freq_in_ms) + 'ms', periods=index_length)

    def _signal_to_256hz_datetimeindexed_series(self, signal_array, signal_type, start_datetime):
        sample_rate = self._raw_data.get_data_channels()[signal_type].get_sample_rate()
        duration_in_seconds = len(signal_array) / float(sample_rate)
        old_sample_points = np.arange(len(signal_array))
        new_sample_points = np.arange(len(signal_array), step=(sample_rate / RATE), dtype=np.double)

        if SPLINES:
            tck = interpolate.splrep(old_sample_points, signal_array)
            resampled_signal_array = interpolate.splev(new_sample_points, tck)
        else:
            raise NotImplementedError('TODO - Non-Splines-Methode implementieren')
            # TODO: - Non-Splines-Methode implementieren
            # resampled_signal_array = signal.resample_poly(
            #     signal_array, RATE, sample_rate, padtype='constant', cval=float('NaN')
            # )

        # tidx = pd.date_range(start_datetime, freq='3.90625ms', periods=len(old_sample_points))
        # ^ NUR WENN OLD_POINTS mit Sample-Rate 256 aufgenommen wurden

        freq_in_ms = 1000/RATE

        idx = pd.date_range(start_datetime, freq=str(freq_in_ms)+'ms', periods=len(new_sample_points))
        resampled_signal_dtseries = pd.Series(resampled_signal_array, index=idx, name=signal_type)


        # plt.plot(tidx, signal_array)
        # plt.plot(idx, resampled_signal_array, c='gold')
        # plt.show()
        return resampled_signal_dtseries


def rms(np_array_like):
    return np.sqrt(np.mean(np.power(np_array_like.astype(np.double), 2)))


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)
