from util.definitions import SLEEP_CLASSIFIERS

FILE_FINDER_IBK = {     #: Maps file type to its file name pattern used to find the file inside the input directory
    'edf': ['*.edf'],
    'sleep_profile': ['*-hypnogram*'],
    'flow_events': ['*-respiration*'],
    'arousals': ['*-arousal*'],
    'event_export': ['*-EventExport*']
}

# python modules
import datetime
from datetime import timedelta

from typing import Tuple, List, Union, Any
import logging

# third-party modules
import pandas as pd

# internal modules
from input_handling import input_reader
from data_structures.raw_data import RawData
from data_structures.raw_data_channel import RawDataChannel
from data_structures.annotation_data import AnnotationData


def read_input(directory_name: str, signals_to_load: List[str] = None,
               read_baseline=True, read_edf_bool=True, read_human_rating=True) -> Tuple[RawData, AnnotationData]:

    filenames = input_reader.find_files(directory_name, FILE_FINDER_IBK.copy())

    annotation_data, start_date = __read_txt_files(filenames)

    if read_edf_bool:
        signals_to_load_ibk = signals_to_load.copy()
        try:
            signals_to_load_ibk.remove('EMG Ment')
            signals_to_load_ibk.append('EMG Ment1')
            signals_to_load_ibk.append('EMG Ment2')
        except ValueError:
            pass

        raw_data: RawData = input_reader.read_edf(filenames['edf'], signals_to_load_ibk.copy())
        raw_data.header['startdate'] = start_date
        try:
            chin_channel_calculated = raw_data.data_channels['EMG Ment1'].get_signal() - raw_data.data_channels['EMG Ment2'].get_signal()
            chin_channel_header = raw_data.data_channels['EMG Ment1'].get_signal_header()
            chin_channel_header['label'] = 'EMG Ment'
            raw_data.add_channel('EMG Ment', RawDataChannel(chin_channel_header, chin_channel_calculated))
            raw_data.data_channels.pop('EMG Ment1')
            raw_data.data_channels.pop('EMG Ment2')
        except KeyError:
            pass

        filenames.pop('edf')
    else:
        raw_data = None

    return raw_data, annotation_data


def __read_annotations(filename, start_datetime, starttime_gap, encoding):

    events_csv = pd.read_csv(filename, sep='\t', header=0, encoding=encoding)

    events = pd.DataFrame()

    events['event_onset'] = (start_datetime - starttime_gap) + pd.to_timedelta(
        events_csv['Start time relative (total µs)'], unit='microseconds'
    )
    events['event_end_time'] = events['event_onset'] + pd.to_timedelta(events_csv['Duration (total µs)'],
                                                                           unit='microseconds')

    events['event'] = events_csv['Subtype']

    return {}, events


def __read_txt_files(filenames) -> tuple[AnnotationData, Union[timedelta, Any]]:
    logging.debug('Start reading .txt files')

    start_datetime, starttime_gap = __find_start_datetime_and_gap(filenames['event_export'], encoding='UTF-16')

    sleep_profile = __read_sleep_profile(filenames['sleep_profile'], start_datetime, starttime_gap, encoding='UTF-16')

    arousals = __read_annotations(filenames['arousals'], start_datetime, starttime_gap, encoding='UTF-16')

    flow_events = __read_annotations(filenames['flow_events'], start_datetime, starttime_gap, encoding='UTF-16')

    return AnnotationData(sleep_profile, flow_events, arousals), start_datetime


def __find_start_datetime_and_gap(filename, encoding):

    event_export = pd.read_csv(filename, header=0, sep='\t', encoding=encoding)
    subtype_speichern = event_export[event_export['Subtype'] == 'Speichern']

    starttime_gap_in_microseconds = int(list(subtype_speichern['Start time relative (total µs)'])[0])
    starttime_gap_timedelta = datetime.timedelta(microseconds=starttime_gap_in_microseconds)

    start_date_string = list(subtype_speichern['Start Date/Time: Date'])[0]
    start_time_string = list(subtype_speichern['Start Date/Time: Time - HH:MM:SS'])[0]
    start_datetime = datetime.datetime.strptime(start_date_string + ',' + start_time_string, '%d.%m.%Y,%H:%M:%S')
    start_datetime = start_datetime + starttime_gap_timedelta

    return start_datetime, starttime_gap_timedelta


def __read_sleep_profile(filename, start_datetime, starttime_gap, encoding):

    hypnogram = pd.read_csv(filename, names=['offset', 'sleep_phase'], sep='\t', header=None, encoding=encoding)

    timestamps = (start_datetime - starttime_gap) + \
                 ((hypnogram['offset']) * datetime.timedelta(seconds=30))
    hypnogram.index = timestamps

    idx = pd.DatetimeIndex(
        pd.date_range(start=start_datetime - starttime_gap,
                      end=timestamps.iloc[-1],
                      freq='30S')
    )
    sleep_profile = pd.DataFrame(index=idx)
    sleep_profile['sleep_phase'] = hypnogram['sleep_phase']

    sleep_profile = sleep_profile.fillna(SLEEP_CLASSIFIERS['artifact'])

    return {}, sleep_profile

