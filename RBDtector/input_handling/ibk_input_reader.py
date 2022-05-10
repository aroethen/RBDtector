FILE_FINDER_IBK = {     #: Maps file type to its file name pattern used to find the file inside the input directory
    'edf': ['*.edf'],
    'sleep_profile': ['*-hypnogram*'],
    'flow_events': ['*-respiration*'],
    'arousals': ['*-arousal*'],
    'event_export': ['*-EventExport*']
}

# python modules
import os
import glob
import datetime
from datetime import date, timedelta

from typing import Tuple, Dict, List, Union, Any
import logging

# third-party modules
from pyedflib import highlevel
import pandas as pd

# internal modules
from input_handling import input_reader
from data_structures.raw_data import RawData
from data_structures.raw_data_channel import RawDataChannel
from data_structures.annotation_data import AnnotationData
from util.error_for_display import ErrorForDisplay
from util import settings


def read_input(directory_name: str, signals_to_load: List[str] = None,
               read_baseline=True, read_edf_bool=True, read_human_rating=True) -> Tuple[RawData, AnnotationData]:

    filenames = input_reader.find_files(directory_name, FILE_FINDER_IBK.copy())

    annotation_data, start_date = __read_txt_files(filenames)

    if read_edf_bool:
        raw_data: RawData = input_reader.read_edf(filenames['edf'], signals_to_load.copy())
        raw_data.header['startdate'] = start_date
        filenames.pop('edf')
    else:
        raw_data = None



    return raw_data, annotation_data


def __read_arousals(param, start_datetime, starttime_gap, encoding):
    pass


def __read_txt_files(filenames) -> tuple[AnnotationData, Union[timedelta, Any]]:
    logging.debug('Start reading .txt files')

    start_datetime, starttime_gap = __find_start_datetime_and_gap(filenames['event_export'], encoding='UTF-16')

    sleep_profile = __read_sleep_profile(filenames['sleep_profile'], start_datetime, starttime_gap, encoding='UTF-16')

    arousals = __read_arousals(filenames['arousals'], start_datetime, starttime_gap, encoding='UTF-16')

    flow_events = None

    # try:
    #     flow_events = __read_flow_events(filenames['flow_events'])
    # except UnicodeDecodeError:
    #     flow_events = __read_flow_events(filenames['flow_events'], encoding='latin-1')
    #
    # try:
    #     arousals = __read_arousals(filenames['arousals'])
    # except UnicodeDecodeError:
    #     arousals = __read_arousals(filenames['arousals'], encoding='latin-1')
    #
    # try:
    #     event_export = __read_event_export(filenames['event_export'])
    # except UnicodeDecodeError:
    #     event_export = __read_event_export(filenames['event_export'], encoding='latin-1')
    #
    # start_date, recording_start_after_midnight = __find_start_date(sleep_profile[0], filenames['sleep_profile'], event_export)
    #
    # logging.debug('.txt files read')
    return AnnotationData(sleep_profile, flow_events, arousals), start_datetime


def __find_start_datetime_and_gap(filename, encoding):

    event_export = pd.read_csv(filename, header=0, sep='\t', encoding=encoding)
    subtype_speichern = event_export[event_export['Subtype'] == 'Speichern']

    starttime_gap_in_microseconds = int(list(subtype_speichern['Start time relative (total µs)'])[0])

    start_date_string = list(subtype_speichern['Start Date/Time: Date'])[0]
    start_time_string = list(subtype_speichern['Start Date/Time: Time - HH:MM:SS'])[0]
    start_datetime = datetime.datetime.strptime(start_date_string + ',' + start_time_string, '%d.%m.%Y,%H:%M:%S')
    start_datetime = start_datetime + datetime.timedelta(microseconds=starttime_gap_in_microseconds)

    return start_datetime, starttime_gap_in_microseconds


def __read_sleep_profile(filename, start_datetime, starttime_gap, encoding):

    hypnogram = pd.read_csv(filename, names=['offset', 'sleep_phase'], sep='\t', header=None, encoding=encoding)

    timestamps = (start_datetime - datetime.timedelta(microseconds=starttime_gap)) + \
                 ((hypnogram['offset'] - 10) * datetime.timedelta(seconds=30))
    hypnogram.index = timestamps

    idx = pd.DatetimeIndex(
        pd.date_range(start=start_datetime - datetime.timedelta(microseconds=starttime_gap),
                      end=timestamps.iloc[-1],
                      freq='30S')
    )
    sleep_profile = pd.DataFrame(index=idx)
    sleep_profile['sleep_phase'] = hypnogram['sleep_phase']

    sleep_profile = sleep_profile.fillna('A')

    return {}, sleep_profile

