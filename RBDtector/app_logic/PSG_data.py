# internal modules
from app_logic.annotation_data import AnnotationData
from app_logic.raw_data import RawData
from input import input_reader as ir
from output import csv_writer

# python modules
import logging
from typing import List

# dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, signal

# DEFINITIONS
SPLINES = True


class PSGData:

    def __init__(self, input_path: str = '', output_path: str = ''):
        self._input_path = input_path
        self._output_path = output_path

        self._raw_data : RawData = None          # content of edf file
        self._annotation_data : AnnotationData = None   # content of txt files
        self._calculated_data : pd.DataFrame = None
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
        self.__read_data()
        self._calculated_data = self.__process_data()
        human_rater_data = self._annotation_data.get_human_rating()
        csv_writer.write_output(self._output_path, human_rating=human_rater_data)

# PRIVATE FUNCTIONS
    def __read_data(self):
        logging.debug('PSGData starting to read input')
        data = ir.read_input(self._input_path)
        self._raw_data = data[0]
        self._annotation_data = data[1]

    def __process_data(self) -> pd.DataFrame:
        # TODO: returns calculated data in form that can be used by output writer
        # for key in self._raw_data.get_data_channels():
        #     print(key)

        df = pd.DataFrame()

        signals_to_evaluate = ['EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.']
        start_datetime = self._raw_data.get_header()['startdate']

        for signal_type in signals_to_evaluate:
            print(signal_type + ' start')
            signal_array = self._raw_data.get_data_channels()[signal_type].get_signal()

            sample_rate = self._raw_data.get_data_channels()[signal_type].get_sample_rate()
            duration_in_seconds = len(signal_array) / float(sample_rate)
            print(str(start_datetime) + ' ' + str(sample_rate) + ' ' + str(duration_in_seconds))

            old_sample_points = np.arange(len(signal_array))
            new_sample_points = np.arange(len(signal_array), step=(sample_rate/200.), dtype=np.double)

            if SPLINES:
                tck = interpolate.splrep(old_sample_points, signal_array)
                resampled_signal_array = interpolate.splev(new_sample_points, tck)
            else:
                resampled_signal_array = signal_array.resample_poly(signal_array, 200, sample_rate, padtype='maximum')

            tidx = pd.date_range(start_datetime, freq='3.90625ms', periods=len(old_sample_points))
            idx = pd.date_range(start_datetime, freq='5ms', periods=len(new_sample_points))
            resampled_signal_dtseries = pd.Series(resampled_signal_array, index=idx, name=signal_type)

            plt.plot(tidx, signal_array)
            plt.plot(idx, resampled_signal_array, c='gold')
            plt.show()

            if df.empty:
                df = resampled_signal_dtseries.to_frame()
            else:
                df[signal_type] = resampled_signal_dtseries

            print(signal_type + ' end')

        print(df)
        # plt.plot(resampled_signal_dtseries.index.values, resampled_signal_dtseries.values)
        # plt.show()


        return None
