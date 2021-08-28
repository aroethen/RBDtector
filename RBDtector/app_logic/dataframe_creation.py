import pandas as pd

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