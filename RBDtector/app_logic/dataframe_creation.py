"""
Dataframe Creation

dataframe_creation groups functions used specifically for creating a
datetime-indexed pandas objects suitable for RBD detection.
"""

import numpy as np
import pandas as pd
from scipy import interpolate, signal

from util import settings


def create_datetime_index(start_datetime, sample_rate, sample_length, index_rate=None):
    """ Creates a datetime index starting at ``start_datetime`` with ``index_rate`` Hz. This translates to a
    frequency of 1000/``index_rate`` milliseconds. The number of index entries are calculated to fit a sample with
    ``sample_length`` recorded at ``sample_rate``.

    :param start_datetime: First entry of datetime index
    :param sample_rate: Recording rate (in Hz) of the sample for which the index will be created
    :param sample_length: Length of the sample for which the index will be created
    :param index_rate: Rate of the datetime index in Hz

    :return: pandas.DatetimeIndex

    """
    if index_rate is None:
        index_rate = settings.RATE
    freq_in_ms = 1000 / index_rate
    index_length = (sample_length / sample_rate) * index_rate

    return pd.date_range(start_datetime, freq=str(freq_in_ms) + 'ms', periods=index_length)


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

    if settings.HZ_1000:
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

        ## FILTER IBK DATA

        b, a = signal.butter(N=4, Wn=50, btype='highpass', fs=hz_rate)
        signal_dtseries.values[:] = signal.filtfilt(b, a, signal_dtseries.values)

        b, a = signal.butter(N=4, Wn=300, btype='lowpass', fs=hz_rate)
        signal_dtseries.values[:] = signal.filtfilt(b, a, signal_dtseries.values)

        b, a = signal.butter(N=4, Wn=[48, 52], btype='bandstop', fs=hz_rate)
        signal_dtseries.values[:] = signal.filtfilt(b, a, signal_dtseries.values)

    else:
        # ## FILTER IBK DATA
        #
        # b, a = signal.butter(N=4, Wn=50, btype='highpass', fs=sample_rate)
        # signal_array[:] = signal.filtfilt(b, a, signal_array)
        #
        # b, a = signal.butter(N=4, Wn=300, btype='lowpass', fs=sample_rate)
        # signal_array[:] = signal.filtfilt(b, a, signal_array)
        #
        # b, a = signal.butter(N=4, Wn=[48, 52], btype='bandstop', fs=sample_rate)
        # signal_array[:] = signal.filtfilt(b, a, signal_array)

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

        # FILTER IBK DATA

        b, a = signal.butter(N=4, Wn=10, btype='highpass', fs=hz_rate)
        signal_dtseries.values[:] = signal.filtfilt(b, a, signal_dtseries.values)

        b, a = signal.butter(N=4, Wn=100, btype='lowpass', fs=hz_rate)
        signal_dtseries.values[:] = signal.filtfilt(b, a, signal_dtseries.values)

        b, a = signal.butter(N=4, Wn=[48, 52], btype='bandstop', fs=hz_rate)
        signal_dtseries.values[:] = signal.filtfilt(b, a, signal_dtseries.values)

    return signal_dtseries
