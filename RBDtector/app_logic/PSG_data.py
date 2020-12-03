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

# DEFINITIONS
SPLINES = True
RATE = 256


class PSGData:

    def __init__(self, input_path: str = '', output_path: str = ''):
        self._input_path = input_path
        self._output_path = output_path

        self._raw_data: RawData = None          # content of edf file
        self._annotation_data: AnnotationData = None   # content of txt files
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
        # for key in self._raw_data.get_data_channels():
        #     print(key)


        signals_to_evaluate = ['EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.']
        start_datetime = self._raw_data.get_header()['startdate']

        # prepare DataFrame with DatetimeIndex
        idx = self._create_datetime_index(start_datetime)
        df = pd.DataFrame(index=idx)

        # add sleep profile to df
        sleep_profile = self._annotation_data.sleep_profile[1]
        print(sleep_profile.index.max() + pd.Timedelta('30s'))

        sleep_profile = sleep_profile.append(pd.DataFrame({'sleep_phase': 'A'},
                                                          index=[sleep_profile.index.max() + pd.Timedelta('30s')])
                                             )

        df = pd.concat([df, sleep_profile.resample(str(1000/RATE)+'ms').ffill()], axis=1, join='inner')
        # df.append(sleep_profile.resample(str(1000/RATE)+'ms').ffill())
        print(df)










        for signal_type in signals_to_evaluate.copy():
            print(signal_type + ' start')

            # Check if signal type exists in edf file
            try:
                signal_array = self._raw_data.get_data_channels()[signal_type].get_signal()
            except KeyError as e:
                signals_to_evaluate.remove(signal_type)
                continue

            # Resample to 256 Hz
            df[signal_type] = self._signal_to_256hz_datetimeindexed_series(signal_array, signal_type,
                                                                                     start_datetime)






            print(signal_type + ' end')

        print(df)
        # plt.plot(resampled_signal_dtseries.index.values, resampled_signal_dtseries.values)
        # plt.show()

        return None

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
