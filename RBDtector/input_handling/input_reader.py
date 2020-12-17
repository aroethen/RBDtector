# python modules
import os
import glob
import datetime

# TODO: Test glob portability to windows - otherwise consider using os.listdir + fnmatch or pathlib.Path().glob
from typing import Tuple, Dict, List
import logging

# third-party modules
from pyedflib import highlevel
import pandas as pd

# internal modules
from data_structures.raw_data import RawData
from data_structures.raw_data_channel import RawDataChannel
from data_structures.annotation_data import AnnotationData
from util.error_for_display import ErrorForDisplay


FILE_FINDER = {
    'edf': '*.edf',
    'sleep_profile': '*Sleep profile*',
    'flow_events': '*Flow Events*',
    'arousals': '*Classification Arousals*',
    'baseline': '*Start-Baseline*',
    'human_rating': '*Generic*'
}


def read_input(directory_name: str) -> Tuple[RawData, AnnotationData]:
    """
    Reads input data from files in given directory into (RawData, AnnotationData)

    :param directory_name: relative or absolute path to input directory
    :returns: Tuple filled with data from the input files in order: (RawData, AnnotationData)
    :raises OSError: if directory_name is not an existing directory
    :raises FileExistsError: if more than one file of a type are found
    :raises FileNotFoundError: if no EDF files are found
    """

    filenames = __find_files(directory_name)
    raw_data: RawData = __read_edf(filenames['edf'])
    del filenames['edf']
    annotation_data: AnnotationData = __read_txt_files(filenames)
    return raw_data, annotation_data


def __find_files(directory_name: str) -> Dict[str, str]:
    """
    Finds EDF and text files in given directory by predefined key words

    :param directory_name: relative or absolute path to input directory
    :returns: dictionary with [edf, sleep_profile, flow_events, arousals, baseline, human_rating]
    as keys and the respective file names as values
    :raises OSError: if directory_name is not an existing directory
    :raises FileExistsError: if more than one file of a type are found
    :raises FileNotFoundError: if no EDF files are found
    """

    files = {}      # return value

    try:
        abs_dir = os.path.abspath(directory_name)

        if not os.path.exists(abs_dir):
            raise ErrorForDisplay('Input directory "{}" does not exist.'.format(directory_name))

    except OSError as e:
        logging.exception()
        raise

    logging.debug('Absolute input directory: ' + abs_dir)

    for file_type, file_identifier in FILE_FINDER.items():
        tmp_files = glob.glob(os.path.join(abs_dir, file_identifier))
        if len(tmp_files) == 1:
            files[file_type] = tmp_files[0]
            logging.debug('{}: {}'.format(file_type, files[file_type]))
        elif len(tmp_files) > 1:
            raise ErrorForDisplay(
                'Too many files of type {} in input directory ({})'.format(file_identifier, abs_dir)
            )

    if 'edf' not in files:
        raise ErrorForDisplay('No EDF files were found in input directory ({}). '
                              'Calculation stopped.'.format(abs_dir))

    return files


def __read_edf(edf: str) -> RawData:
    """
    Reads an .edf file using pyEDFlib and stores its data inside a RawData return object
    :param edf: path to .edf file
    :returns: RawData object containing all information of .edf file
    """
    logging.debug('Start reading .edf files')

    signals, signal_headers, header = highlevel.read_edf(edf)

    if len(signals) != len(signal_headers):
        raise ValueError('Input .edf file has a different amount of signal_headers and signals.')

    data_channels = {}

    for signal_header, signal in zip(signal_headers, signals):
        data_channels[signal_header['label']] = RawDataChannel(signal_header, signal)

    logging.debug('.edf files read')

    return RawData(header, data_channels)


def __read_txt_files(filenames: Dict[str, str]) -> AnnotationData:
    """
    Reads data of all sleep
    :param filenames:
    :return:
    """
    logging.debug('Start reading .txt files')

    sleep_profile = __read_sleep_profile(filenames['sleep_profile'])
    flow_events = __read_flow_events(filenames['flow_events'])
    arousals = __read_arousals(filenames['arousals'])
    baseline = __read_baseline(filenames['baseline'],
                               start_date=__find_start_date(sleep_profile[0], filenames['sleep_profile']))
    human_rating = __read_human_rating(filenames['human_rating'])

    logging.debug('.txt files read')
    return AnnotationData(sleep_profile, flow_events, arousals, baseline, human_rating)


def __read_sleep_profile(filename: str) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Reads text file of sleep profile at file path 'filename'
    Text file format:
        - header:
            - keys that are not numbers
            - keys and values separated by first ':' in line
        - data:
            - all lines in format: %H:%M:%S,%f; <sleeping phase as string>
    :param filename: path of sleep profile text file
    :return: Tuple containing the header as dictionary and a pandas DataFrame containing time stamps and classification
    """
    header: [Dict[str, str]] = {}
    df: pd.DataFrame = None

    with open(filename, 'r', encoding='utf-8') as f:

        text_in_lines = f.readlines()
        timestamps: List[pd.Timestamp] = []
        sleep_events: List[str] = []
        header, first_line_of_data = __read_annotation_header(text_in_lines)
        start_date = __find_start_date(header, filename)

        # Loop over timestamps and events and read them into timestamps and sleep_events
        current_date = start_date
        date_change_occurred = False

        for line in text_in_lines[first_line_of_data:]:
            time, _, sleep_event = line.partition(';')

            if not date_change_occurred \
                    and datetime.time(0, 0, 0) <= datetime.datetime.strptime(time, '%H:%M:%S,%f').time() < datetime.time(12, 0, 0):
                current_date = start_date + datetime.timedelta(days=1)
                date_change_occurred = True

            current_date_time = pd.to_datetime(str(current_date) + ' ' + time, infer_datetime_format=True)

            timestamps.append(current_date_time)
            sleep_events.append(sleep_event.strip())

        # create DataFrame with columns timestamps and sleep_events
        df = pd.DataFrame(
            {
                'sleep_phase': sleep_events
            }, index=timestamps)
        df['sleep_phase'].astype('category')

        f.close()
    return header, df


def __read_flow_events(filename: str) -> Tuple[Dict[str, str], pd.DataFrame]:

    with open(filename, 'r', encoding='utf-8') as f:
        text_in_lines = f.readlines()
        header, first_line_of_data = __read_annotation_header(text_in_lines)
        start_date = __find_start_date(header, filename)

        event_onsets = []
        event_end_times = []
        durations_in_seconds = []
        flow_events = []

        # Loop over times, durations and events and read them into respective lists
        for line in text_in_lines[first_line_of_data:]:
            split_list = line.split('-', 1)
            event_onset = split_list[0].strip()
            split_list2 = str(split_list[1]).split(';')
            event_end_time = split_list2[0].strip()
            duration = split_list2[1].strip()
            flow_event = split_list2[2].strip()

            onset_after_midnight = 0
            end_after_midnight = 0

            if datetime.time(0, 0, 0) <= datetime.datetime.strptime(event_onset, '%H:%M:%S,%f').time() < datetime.time(12, 0, 0):
                onset_after_midnight = 1
            if datetime.time(0, 0, 0) <= datetime.datetime.strptime(event_end_time, '%H:%M:%S,%f').time() < datetime.time(12, 0, 0):
                end_after_midnight = 1

            onset_timestamp = pd.to_datetime(str(start_date + datetime.timedelta(days=onset_after_midnight))
                                             + ' ' + event_onset, infer_datetime_format=True)
            end_timestamp = pd.to_datetime(str(start_date + datetime.timedelta(days=end_after_midnight))
                                           + ' ' + event_end_time, infer_datetime_format=True)

            event_onsets.append(onset_timestamp)
            event_end_times.append(end_timestamp)
            durations_in_seconds.append(duration)
            flow_events.append(flow_event)

        # create DataFrame with filled lists as columns
        df = pd.DataFrame(
            {
                'event_onset': event_onsets,
                'event_end_time': event_end_times,
                'duration_in_seconds': durations_in_seconds,
                'event': flow_events
            })
        df['event'].astype('category')

        f.close()

        return header, df


def __read_arousals(filename: str) -> Tuple[Dict[str, str], pd.DataFrame]:

    with open(filename, 'r', encoding='utf-8') as f:
        text_in_lines = f.readlines()
        header, first_line_of_data = __read_annotation_header(text_in_lines)
        start_date = __find_start_date(header, filename)

        event_onsets = []
        event_end_times = []
        durations_in_seconds = []
        arousals = []

        # Loop over times, durations and events and read them into respective lists
        for line in text_in_lines[first_line_of_data:]:
            split_list = line.split('-', 1)
            event_onset = split_list[0].strip()
            split_list2 = str(split_list[1]).split(';')
            event_end_time = split_list2[0].strip()
            duration = split_list2[1].strip()
            arousal = split_list2[2].strip()

            onset_after_midnight = 0
            end_after_midnight = 0

            if datetime.time(0, 0, 0) <= datetime.datetime.strptime(event_onset, '%H:%M:%S,%f').time() < datetime.time(12, 0, 0):
                onset_after_midnight = 1
            if datetime.time(0, 0, 0) <= datetime.datetime.strptime(event_end_time, '%H:%M:%S,%f').time() < datetime.time(12, 0, 0):
                end_after_midnight = 1

            onset_timestamp = pd.to_datetime(str(start_date + datetime.timedelta(days=onset_after_midnight))
                                             + ' ' + event_onset, infer_datetime_format=True)
            end_timestamp = pd.to_datetime(str(start_date + datetime.timedelta(days=end_after_midnight))
                                           + ' ' + event_end_time, infer_datetime_format=True)

            event_onsets.append(onset_timestamp)
            event_end_times.append(end_timestamp)
            durations_in_seconds.append(duration)
            arousals.append(arousal)

        # create DataFrame with filled lists as columns
        df = pd.DataFrame(
            {
                'event_onset': event_onsets,
                'event_end_time': event_end_times,
                'duration_in_seconds': durations_in_seconds,
                'event': arousals
            })
        df['event'].astype('category')

        f.close()

    return header, df


def __read_baseline(filename: str, start_date: datetime.date) -> Dict[str, datetime.datetime]:
    with open(filename, 'r', encoding='utf-8') as f:
        text_in_lines = f.readlines()
        baseline_dict, _ = __read_annotation_header(text_in_lines)

        for key, value in baseline_dict.items():

            if value.lower() == 'none':
                baseline_dict[key] = ''
                continue
            else:

                value = value.split('-')

                if isinstance(value, list):
                    for i, time_string in enumerate(value):

                        time_string = time_string.replace('.', ':')

                        after_midnight = 0
                        if datetime.time(0, 0, 0) <= datetime.datetime.strptime(time_string, '%H:%M:%S').time() < datetime.time(12, 0, 0):
                            after_midnight = 1

                        value[i] = pd.to_datetime(str(start_date + datetime.timedelta(days=after_midnight))
                                                  + ' ' + time_string, infer_datetime_format=True)
                    baseline_dict[key] = value

        return baseline_dict


def __read_human_rating(filename: str) -> Tuple[Dict[str, str], pd.DataFrame]:

    with open(filename, 'r', encoding='utf-8') as f:
        text_in_lines = f.readlines()
        header, first_line_of_data = __read_annotation_header(text_in_lines)
        start_date = __find_start_date(header, filename)

        event_onsets = []
        event_end_times = []
        events = []

        # Loop over times, durations and events and read them into respective lists
        for line in text_in_lines[first_line_of_data:]:
            split_list = line.split('-', 1)
            event_onset = split_list[0].strip()
            split_list2 = str(split_list[1]).split(';')
            event_end_time = split_list2[0].strip()
            event = split_list2[1].strip()

            onset_after_midnight = 0
            end_after_midnight = 0

            if datetime.time(0, 0, 0) <= datetime.datetime.strptime(event_onset, '%H:%M:%S,%f').time() < datetime.time(12, 0, 0):
                onset_after_midnight = 1
            if datetime.time(0, 0, 0) <= datetime.datetime.strptime(event_end_time, '%H:%M:%S,%f').time() < datetime.time(12, 0, 0):
                end_after_midnight = 1

            onset_timestamp = pd.to_datetime(str(start_date + datetime.timedelta(days=onset_after_midnight))
                                             + ' ' + event_onset, infer_datetime_format=True)
            end_timestamp = pd.to_datetime(str(start_date + datetime.timedelta(days=end_after_midnight))
                                           + ' ' + event_end_time, infer_datetime_format=True)

            event_onsets.append(onset_timestamp)
            event_end_times.append(end_timestamp)
            events.append(event)

        # create DataFrame with filled lists as columns
        df = pd.DataFrame(
            {
                'event_onset': event_onsets,
                'event_end_time': event_end_times,
                'event': events
            })
        df['event'].astype('category')

        f.close()

        return header, df


def __read_annotation_header(text_in_lines: List[str]) -> Tuple[Dict[str, str], int]:
    """
    Read header data of annotation data from input text file into a dictionary
    :param text_in_lines: Output from file.readlines() of annotation text file
    :return: Dictionary containing header data and number of first line of data
    """

    header: [Dict[str, str]] = {}
    first_line_of_data: int = 0

    # Read header data and find first line of data
    for index, line in enumerate(text_in_lines):

        key, separator, value = line.partition(':')

        if separator == '':
            continue

        if not key.isdecimal():
            header[key.strip()] = value.strip()
        else:
            first_line_of_data = index
            break

    return header, first_line_of_data


def __find_start_date(header: Dict[str, str], filename: str) -> datetime.date:
    """
    Takes start time string out of dictionary['Start Time'], extracts the date part and returns it as datetime.date
    :param header: Dictionary containing a key 'Start Time' with date and time in a string as value
    :param filename: filename of input file. Used for exception handling only.
    :return: Start date as date object
    """
    try:
        start_time = pd.Timestamp(header['Start Time'])
        start_date = start_time.date()

    except KeyError as e:
        logging.exception()

        raise ErrorForDisplay('"Start Time" field missing in header of the following file:' + filename) from e

    return start_date


if __name__ == '__main__':
    pass
