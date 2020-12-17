# internal modules
from data_structures.annotation_data import AnnotationData
from data_structures.raw_data import RawData
from input_handling import input_reader as ir
from output import csv_writer

# python modules
import logging

# dependencies
import numpy as np
import pandas as pd
from scipy import interpolate, signal

# dev dependencies
import matplotlib.pyplot as plt
import os.path

# DEFINITIONS
SPLINES = True
RATE = 256
FLOW = True
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
    'AUX': 'LeftArm',       #TODO: VERIFY ARMS LEFT/RIGHT MAPPING
    'Akti.': 'RightArm'
}

EVENT_TYPE = {
    'tonic': 'Tonic',
    'intermediate': 'Any',
    'phasic': 'Phasic',
    'artefact': 'Artifact'
}


class PSGData:

    def __init__(self, input_path: str = '', output_path: str = ''):
        self._input_path = input_path
        self._output_path = output_path

        self._raw_data: RawData                  # content of edf file
        self._annotation_data: AnnotationData    # content of txt files
        self._calculated_data: pd.DataFrame = None
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
        logging.debug('PSGData starting to generate output')
        self._read_data()
        self._calculated_data = self._process_data()
        human_rater_data = self._annotation_data.get_human_rating()
        csv_writer.write_output(self._output_path, human_rating=human_rater_data)

# PRIVATE FUNCTIONS
    def _read_data(self):
        logging.debug('PSGData starting to read input')
        data = ir.read_input(self._input_path)
        self._raw_data = data[0]
        self._annotation_data = data[1]

    def _process_data(self) -> pd.DataFrame:
        # TODO: returns calculated data in form that can be used by output writer

        signals_to_evaluate = ['EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.']
        start_datetime = self._raw_data.get_header()['startdate']

        if DEV:
            df = pd.read_pickle(os.path.join(self.output_path, 'pickledDF'))
        else:
            # prepare DataFrame with DatetimeIndex
            idx = self._create_datetime_index(start_datetime)
            df = pd.DataFrame(index=idx)

            # add sleep profile to df
            df = self.add_sleep_profile_to_df(df)

            # add artefacts to df
            df = self.add_artefacts_to_df(df)

            # pickle for further DEV use as long as adding artefacts into df is time consuming
            df.to_pickle(os.path.join(self.output_path, 'pickledDF'))

        # process human rating for evaluation per signal and event
        human_rating = self._annotation_data.human_rating[1]
        human_rating_label_dict = human_rating.groupby('event').groups
        print(human_rating_label_dict)

        # FOR EACH EMG SIGNAL:
        for signal_type in signals_to_evaluate.copy():
            print(signal_type + ' start')

            # Check if signal type exists in edf file
            try:
                signal_array = self._raw_data.get_data_channels()[signal_type].get_signal()
            except KeyError as e:
                signals_to_evaluate.remove(signal_type)
                continue

            # Resample to 256 Hz
            df[signal_type] = self._signal_to_256hz_datetimeindexed_series(signal_array, signal_type, start_datetime)

            # add signal type baseline column
            df = self.add_signal_baseline_to_df(df, signal_type)

            # find all samples with at least 2x baseline in REM
            df[signal_type + '_isGE2xBaseline'] = df[signal_type].abs() >= 2 * df[signal_type + '_baseline']
            # print(df[signal_type].where(df[signal_type + '_isGE2xBaseline']).dropna())

            # add human rating boolean arrays

            # Human tonic columns for signal type and respective events
            # df[signal_type + '_human_tonic'] = pd.Series(False, index=df.index)
            #
            # df[signal_type + '_human_intermediate'] = pd.Series(False, index=df.index)

            for event_type in EVENT_TYPE.keys():
                print('Event type: ' + event_type)
                df[signal_type + '_human_' + event_type] = pd.Series(False, index=df.index)

                print(HUMAN_RATING_LABEL[signal_type] + EVENT_TYPE[event_type])
                human_event_type_indices = \
                    human_rating_label_dict.get(HUMAN_RATING_LABEL[signal_type] + EVENT_TYPE[event_type], [])
                print(human_event_type_indices)
                for idx in human_event_type_indices:
                    print(human_rating.iloc[idx]['event_onset'])
                    df.loc[
                        human_rating.iloc[idx]['event_onset']:human_rating.iloc[idx]['event_end_time'],
                        [signal_type + '_human_' + event_type]
                    ] = True

            print(signal_type + ' end')

        print(df.info())
        # REM PHASES
        plt.fill_between(df.index.values, df['is_REM']*(-1000), df['is_REM']*1000,
                         facecolor='lightsteelblue', label="is_REM")

        # HUMAN RATING
        plt.fill_between(df.index.values, df['EMG_human_tonic']*(-1000), df['EMG_human_tonic']*1000,
                         facecolor='mediumorchid', label="EMG_human_tonic")
        plt.fill_between(df.index.values, df['EMG_human_phasic']*(-1000), df['EMG_human_phasic']*1000,
                         facecolor='deeppink', label="EMG_human_phasic")
        plt.fill_between(df.index.values, df['EMG_human_intermediate']*(-1000), df['EMG_human_intermediate']*1000,
                         facecolor='violet', label="EMG_human_intermediate")
        plt.fill_between(df.index.values, df['EMG_human_artefact']*(-1000), df['EMG_human_artefact']*1000,
                         facecolor='maroon', label="EMG_human_artefact")

        # plt.fill_between(df.index.values, 0, , facecolor='lightsteelblue')
        plt.plot(df.index.values, df['EMG'], c='#313133', label="EMG")
        plt.plot(df['EMG_baseline'], c='mediumseagreen', label="EMG_baseline")
        plt.plot(df['EMG_baseline']*(-1), c='mediumseagreen')
        plt.plot([df.index.values[0], df.index.values[-1]], [0, 0], c='dimgrey')
        plt.scatter(df.index.values, df['EMG'].where(df['EMG' + '_isGE2xBaseline']), s=4, c='lime',
                    label='EMG GE 2x Baseline')

        plt.fill_between(df.index.values, df['is_artefact'] * (-200), df['is_artefact'] * 200,
                         facecolor='dimgrey', label="is_artefact", alpha=0.7, zorder=10)
        plt.legend(loc='upper right')
        plt.show()


        # plt.plot(df['artefact_event'])
        # # plt.plot(df['is_artefact'], label='is_artefact')
        # plt.plot(df['is_REM'] * (-1), label='is_REM')
        # plt.legend(loc='lower right')
        # plt.show()

        # plt.plot(resampled_signal_dtseries.index.values, resampled_signal_dtseries.values)
        # plt.show()

        return None

    def add_signal_baseline_to_df(self, df, signal_type):
        bl_start, bl_stop = self._annotation_data.baseline[BASELINE_NAME[signal_type]]
        baseline = rms(df.loc[bl_start: bl_stop, signal_type])
        df[signal_type + '_baseline'] = baseline
        return df

    def add_artefacts_to_df(self, df):
        # TODO: OPTIMIZE FOR SPEED!!!

        # get arousals, add them as string to 'artefact_event' column and mark as 'arousal' in 'artefact_type' column
        arousals: pd.DataFrame = self._annotation_data.arousals[1]
        df['artefact_event'] = ''
        for label, on, off in zip(arousals['event'], arousals['event_onset'], arousals['event_end_time']):
            # df.loc[on:off, ['artefact_event']].str.cat += label
            # df.loc[on:off, ['artefact_event']] = df.artefact_event[on:off].apply(lambda x: '#'.join([x, label]))
            df.loc[on:off, ['artefact_event']] = df.loc[on:off, ['artefact_event']] + label
            # df.update(df.loc[on:off, ['artefact_event']].applymap(lambda x: '#'.join([x, label])))

        if FLOW:
            # get flow events, add them as string to 'artefact_event' column and mark as 'flow' in 'artefact_type'
            # column make that part optional via FLOW "macro"
            flow_events = self._annotation_data.flow_events[1]
            df['flow_event'] = ''
            for label, on, off in zip(flow_events['event'], flow_events['event_onset'], flow_events['event_end_time']):
                df.loc[on:off, ['flow_event']] = df.loc[on:off, ['flow_event']] + label

        # add conditional column 'is_artefact'
        if FLOW:
            df['is_artefact'] = np.logical_or(df['artefact_event'].astype(bool), df['flow_event'].astype(bool))
        else:
            df['is_artefact'] = df['artefact_event'].astype(bool)

        return df

    def add_sleep_profile_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        sleep_profile: pd.DataFrame = self._annotation_data.sleep_profile[1]
        # print(sleep_profile.index.max() + pd.Timedelta('30s'))

        # append a final row to sleep profile with "Artifact" sleep phase
        # for easier concatenation with df while correctly accounting for last 30s sleep profile interval
        sleep_profile = sleep_profile.append(pd.DataFrame({'sleep_phase': 'A'},
                                                          index=[sleep_profile.index.max() + pd.Timedelta('30s')])
                                             )
        # resample sleep profile from 2Hz(30s intervals) to 256 Hz, fill all entries with the correct sleeping phase
        # and add it as column to dataframe
        df = pd.concat([df, sleep_profile.resample(str(1000 / RATE) + 'ms').ffill()], axis=1, join='inner')
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
        print(str(start_datetime) + ' ' + str(sample_rate) + ' ' + str(duration_in_seconds))
        old_sample_points = np.arange(len(signal_array))
        new_sample_points = np.arange(len(signal_array), step=(sample_rate / RATE), dtype=np.double)

        if SPLINES:
            tck = interpolate.splrep(old_sample_points, signal_array)
            resampled_signal_array = interpolate.splev(new_sample_points, tck)
        else:
            raise NotImplementedError('TODO - Non-Splines-Methode implementieren')
            #TODO: - Non-Splines-Methode implementieren
            resampled_signal_array = signal.resample_poly(
                signal_array, RATE, sample_rate, padtype='constant', cval=float('NaN')
            )

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
