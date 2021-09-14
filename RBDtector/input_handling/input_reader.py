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
from util.definitions import FILE_FINDER


def read_input(directory_name: str, signals_to_load: List[str] = None, read_baseline=True, read_edf=True) -> Tuple[RawData, AnnotationData]:
    """
    Reads input data from files in given directory into (RawData, AnnotationData)

    :param read_edf:
    :param directory_name: relative or absolute path to input directory
    :param signals_to_load: a list of strings containing all signal names to be loaded from edf file.
                Passing None results in all signals being loaded. Defaults to None.
    :param read_baseline: boolean value whether to read baseline data from file

    :returns: Tuple filled with data from the input files in order: (RawData, AnnotationData)

    :raises OSError: if directory_name is not an existing directory
    :raises FileExistsError: if more than one file of a type are found
    :raises FileNotFoundError: if no EDF files are found
    """
    filenames = __find_files(directory_name)

    if read_edf:
        raw_data: RawData = __read_edf(filenames['edf'], signals_to_load)
        filenames.pop('edf')
    else:
        raw_data = None

    annotation_data: AnnotationData = __read_txt_files(filenames, read_baseline)
    return raw_data, annotation_data


def __find_files(directory_name: str, find_annotation_only=False) -> Dict[str, str]:
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

    logging.info('Absolute input directory: ' + abs_dir)

    for file_type, file_identifier in FILE_FINDER.items():
        tmp_files = glob.glob(os.path.join(abs_dir, file_identifier))
        if len(tmp_files) == 1:
            if file_type == 'human_rating':
                files[file_type] = tmp_files
                logging.info('{}: {}'.format(file_type, files[file_type]))
            else:
                files[file_type] = tmp_files[0]
                logging.info('{}: {}'.format(file_type, files[file_type]))

        elif len(tmp_files) > 1:
            if file_type == 'human_rating':
                tmp_files.sort()
                if "NO" not in tmp_files[1]:
                    tmp_files[0], tmp_files[1] = tmp_files[1], tmp_files[0]

                files[file_type] = tmp_files
                logging.info('{}: {}'.format(file_type, files[file_type]))
            else:
                raise ErrorForDisplay(
                    'Too many files of type {} in input directory ({})'.format(file_identifier, abs_dir)
                )

    if 'edf' not in files:
        if not find_annotation_only:
            raise ErrorForDisplay('No EDF files were found in input directory ({}). '
                                  'Calculation stopped.'.format(abs_dir))
    else:
        if find_annotation_only:
            files.pop('edf')

    return files


def __read_edf(edf: str, signals_to_load: List[str] = None) -> RawData:
    """
    Reads an .edf file using pyEDFlib and stores its data inside a RawData return object
    :param edf: path to .edf file
    :param signals_to_load: a list of strings containing all signal names to be loaded from edf file.
                Passing None results in all signals being loaded. Defaults to None.
    :returns: RawData object containing all information of .edf file
    """
    logging.debug('Start reading .edf files')

    signals, signal_headers, header = highlevel.read_edf(edf, ch_names=signals_to_load)   # TODO: Only load needed signals!!!

    if len(signals) != len(signal_headers):
        raise ValueError('Input .edf file has a different amount of signal_headers and signals.')

    data_channels = {}

    for signal_header, signal in zip(signal_headers, signals):
        data_channels[signal_header['label']] = RawDataChannel(signal_header, signal)

    logging.debug('.edf files read')

    return RawData(header, data_channels)


def __read_txt_files(filenames: Dict, read_baseline: bool = True) -> AnnotationData:
    """
    Reads data of all polysomnography annotation files
    :param filenames:
    :return:
    """
    logging.debug('Start reading .txt files')

    sleep_profile = __read_sleep_profile(filenames['sleep_profile'])
    flow_events = __read_flow_events(filenames['flow_events'])
    arousals = __read_arousals(filenames['arousals'])

    start_date, recording_start_after_midnight = __find_start_date(sleep_profile[0], filenames['sleep_profile'])
    if read_baseline:
        baseline = __read_baseline(
            filenames['baseline'], start_date=start_date, recording_start_after_midnight=recording_start_after_midnight)
    else:
        baseline = None

    human_rating = __read_human_rating(
        filenames['human_rating'], start_date=start_date, recording_start_after_midnight=recording_start_after_midnight)

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
        start_date, recording_start_after_midnight = __find_start_date(header, filename)

        # Loop over timestamps and events and read them into timestamps and sleep_events
        current_date = start_date
        date_change_occurred = False

        for line in text_in_lines[first_line_of_data:]:
            time, _, sleep_event = line.partition(';')

            if not recording_start_after_midnight \
                    and not date_change_occurred \
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
        start_date, recording_start_after_midnight = __find_start_date(header, filename)

        # event name stands on 'event_name_split_index'th position after timings in annotation file
        event_name_split_index = 2

        event_onsets, event_name_list, event_end_times = \
            __read_annotation_body(text_in_lines[first_line_of_data:], event_name_split_index, start_date,
                                   recording_start_after_midnight)

        # create DataFrame with filled lists as columns
        df = pd.DataFrame(
            {
                'event_onset': event_onsets,
                'event_end_time': event_end_times,
                'event': event_name_list
            })
        df['event'].astype('category')

        f.close()

        return header, df


def __read_arousals(filename: str) -> Tuple[Dict[str, str], pd.DataFrame]:

    with open(filename, 'r', encoding='utf-8') as f:
        text_in_lines = f.readlines()
        header, first_line_of_data = __read_annotation_header(text_in_lines)
        start_date, recording_start_after_midnight = __find_start_date(header, filename)

        # event name stands on 'event_name_split_index'th position after timings in annotation file
        event_name_split_index = 2

        event_onsets, event_name_list, event_end_times = __read_annotation_body(text_in_lines[first_line_of_data:],
                                                                                event_name_split_index, start_date,
                                                                                recording_start_after_midnight)

        # create DataFrame with filled lists as columns
        df = pd.DataFrame(
            {
                'event_onset': event_onsets,
                'event_end_time': event_end_times,
                'event': event_name_list
            })
        df['event'].astype('category')

        f.close()

    return header, df


def __read_baseline(filename: str, start_date: datetime.date, recording_start_after_midnight) -> Dict[str, datetime.datetime]:
    try:
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

                            time_string = time_string.replace('.', ':').strip()

                            if not recording_start_after_midnight:
                                after_midnight = 0
                                if datetime.time(0, 0, 0) <= datetime.datetime.strptime(time_string, '%H:%M:%S').time() < datetime.time(12, 0, 0):
                                    after_midnight = 1

                                value[i] = pd.to_datetime(str(start_date + datetime.timedelta(days=after_midnight))
                                                          + ' ' + time_string, infer_datetime_format=True)

                            else:
                                value[i] = pd.to_datetime(str(start_date + datetime.timedelta(days=0))
                                                          + ' ' + time_string, infer_datetime_format=True)

                        baseline_dict[key] = value

        return baseline_dict
    except OSError as e:
        logging.exception(e)
        raise ErrorForDisplay('Baseline file: "' + filename + '" could not be opened') from e
    except Exception as e:
        logging.exception(e)
        raise ErrorForDisplay('An error occurred while parsing the baseline file: "'
                              + filename +
                              '\nPlease check for errors inside the file.'
                              '\nFull traceback information of the error is logged in logfile.txt.') from e


def __read_human_rating(filenames: List[str], start_date, recording_start_after_midnight) -> Tuple[Dict[str, str], pd.DataFrame]:
    human_rating = []

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            text_in_lines = f.readlines()
            header, first_line_of_data = __read_annotation_header(text_in_lines)

            # event name stands on 'event_name_split_index'th position after timings in annotation file
            event_name_split_index = 1

            event_onsets, event_name_list, event_end_times = \
                __read_annotation_body(text_in_lines[first_line_of_data:], event_name_split_index, start_date,
                                       recording_start_after_midnight)

            # create DataFrame with filled lists as columns
            df = pd.DataFrame(
                {
                    'event_onset': event_onsets,
                    'event_end_time': event_end_times,
                    'event': event_name_list
                })
            df['event'].astype('category')
            human_rating.append((header, df))
            f.close()

    return human_rating


def __read_annotation_header(text_in_lines: List[str]) -> Tuple[Dict[str, str], int]:
    """
    Read header data of annotation data from input text file into a dictionary
    :param text_in_lines: Output from file.readlines() of annotation text file
    :return: Dictionary containing header data and index of first line of data
    """

    header: [Dict[str, str]] = {}
    first_line_of_data: int = -1

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


def __read_annotation_body(annotation_body_in_lines, event_name_split_index, start_date, recording_start_after_midnight):
    """
    Read the event onsets, event end times and event names out of 'annotation_body_in_lines', with start_date being the
    date of the beginning of the recording and 'event_name_split_index' the position of the event_name after the
    timestamps.
    :param annotation_body_in_lines: lines of annotation body text
    :param event_name_split_index: position after the timestamps of the event_name in a line of the annotation body
    :param start_date: date of the beginning of the recording
    :return event_onsets, event_name_list, event_end_times: Lists of respective event_onsets, end_times and event_names
    of the events in the annotation body
    """

    event_onsets = []
    event_end_times = []
    event_name_list = []
    # Loop over times, durations and events and read them into respective lists
    for line in annotation_body_in_lines:
        if line.strip() == "":
            continue

        split_list = line.split('-', 1)
        event_onset = split_list[0].strip()
        split_list2 = str(split_list[1]).split(';')
        event_end_time = split_list2[0].strip()
        event_name = split_list2[event_name_split_index].strip()

        if not recording_start_after_midnight:
            onset_after_midnight = 0
            end_after_midnight = 0

            if datetime.time(0, 0, 0) <= datetime.datetime.strptime(event_onset, '%H:%M:%S,%f').time() < \
                    datetime.time(12, 0, 0):
                onset_after_midnight = 1
            if datetime.time(0, 0, 0) <= datetime.datetime.strptime(event_end_time, '%H:%M:%S,%f').time() < datetime.time(
                    12, 0, 0):
                end_after_midnight = 1

            onset_timestamp = pd.to_datetime(str(start_date + datetime.timedelta(days=onset_after_midnight))
                                             + ' ' + event_onset, infer_datetime_format=True)
            end_timestamp = pd.to_datetime(str(start_date + datetime.timedelta(days=end_after_midnight))
                                           + ' ' + event_end_time, infer_datetime_format=True)

        else:  # if recording_start_after_midnight

            onset_timestamp = pd.to_datetime(str(start_date + datetime.timedelta(days=0))
                                             + ' ' + event_onset, infer_datetime_format=True)
            end_timestamp = pd.to_datetime(str(start_date + datetime.timedelta(days=0))
                                           + ' ' + event_end_time, infer_datetime_format=True)

        event_onsets.append(onset_timestamp)
        event_end_times.append(end_timestamp)
        event_name_list.append(event_name)

    return event_onsets, event_name_list, event_end_times

def __find_start_date(header: Dict[str, str], filename: str) -> datetime.date:
    """
    Takes start time string out of dictionary['Start Time'], extracts the date part and returns it as datetime.date
    :param header: Dictionary containing a key 'Start Time' with date and time in a string as value
    :param filename: filename of input file. Used for exception handling only.
    :return: Start date as date object
    """
    try:
        start_time = pd.Timestamp(datetime.datetime.strptime(header['Start Time'], '%d.%m.%Y %H:%M:%S'))
        start_date = start_time.date()
        recording_start_after_midnight = (datetime.time(0, 0, 0) <= start_time.time() < datetime.time(12, 0, 0))

    except KeyError as e:
        logging.exception()

        raise ErrorForDisplay('"Start Time" field missing in header of the following file:' + filename) from e

    return start_date, recording_start_after_midnight


if __name__ == '__main__':
    pass
