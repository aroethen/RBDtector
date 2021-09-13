import numpy as np
import pandas as pd
from scipy import interpolate

from util.settings import Settings


class DataframeCreation:
    """ DataframeCreation serves as a namespace. It groups all functions used specifically for creating
    datetime-indexed pandas objects suitable for RBD detection. """

    @staticmethod
    def create_datetime_index(start_datetime, sample_rate, sample_length, index_rate=Settings.RATE):
        """ Creates a datetime index starting at ``start_datetime`` with ``index_rate`` Hz. This translates to a
        frequency of 1000/``index_rate`` milliseconds. The number of index entries are calculated to fit a sample with
        ``sample_length`` recorded at ``sample_rate``.

        :param start_datetime: First entry of datetime index
        :param sample_rate: Recording rate (in Hz) of the sample for which the index will be created
        :param sample_length: Length of the sample for which the index will be created
        :param index_rate: Rate of the datetime index in Hz

        :return: pandas.DatetimeIndex

        """
        freq_in_ms = 1000 / index_rate
        index_length = (sample_length / sample_rate) * index_rate

        return pd.date_range(start_datetime, freq=str(freq_in_ms) + 'ms', periods=index_length)

    @staticmethod
    def signal_to_hz_rate_datetimeindexed_series(hz_rate, sample_rate, signal_array, signal_type, start_datetime):
        """ Transform signal_array to pandas Series with datetime-index starting from 'start_datetime'
        with sample rate 'hz_rate'. If the sample rate of the signal array does not match the desired Hz rate it is
        resampled using spline interpolation.
        :param hz_rate: Desired rate for pandas.DatetimeIndex in Hertz
        :param sample_rate: Sample rate of 'signal_array'
        :param signal_array: (Numpy) array with floating point signal of EMG channel
        :param signal_type: EMG channel name of 'signal_array'
        :param start_datetime: start date and time for pandas.DatetimeIndex
        :return: Signal as a datetimeindexed pd.Series starting at 'start_datetime' with sample rate 'hz_rate'
        """
        freq_in_ms = 1000 / hz_rate

        if hz_rate != sample_rate:
            # resample to hz_rate if necessary
            old_sample_points = np.arange(len(signal_array))
            new_sample_points = np.arange(len(signal_array), step=(sample_rate / hz_rate), dtype=np.double)

            tck = interpolate.splrep(old_sample_points, signal_array)
            resampled_signal_array = interpolate.splev(new_sample_points, tck)

            idx = pd.date_range(start_datetime, freq=str(freq_in_ms)+'ms', periods=len(new_sample_points))
            signal_dtseries = pd.Series(resampled_signal_array, index=idx, name=signal_type)

        else:
            idx = pd.date_range(start_datetime, freq=str(freq_in_ms) + 'ms', periods=len(signal_array))
            signal_dtseries = pd.Series(signal_array, index=idx, name=signal_type)

        return signal_dtseries
