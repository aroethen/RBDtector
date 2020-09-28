# python modules
import os
import glob
# TODO: Test glob portability to windows - otherwise consider using os.listdir + fnmatch or pathlib.Path().glob
from typing import Tuple, Dict
import logging

# third-party modules
from pyedflib import highlevel
import numpy as np
import pandas as pd

# internal modules
from app_logic.raw_data import RawData
from app_logic.raw_data_channel import RawDataChannel
from app_logic.annotation_data import AnnotationData
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
            raise FileExistsError(
                'Too many files of type {} in input directory ({})'.format(file_identifier, abs_dir)
            )

    if 'edf' not in files:
        raise FileNotFoundError('No EDF files were found in input directory ({}). '
                                'Calculation stopped.'
                                .format(abs_dir))

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
    logging.debug('Start reading .txt files')

    sleep_profile = __read_sleep_profile(filenames['sleep_profile'])
    # flow_events = __read_flow_events(filenames['flow_events'])
    # arousals = __read_arousals(filenames['arousals'])
    # baseline = __read_baseline(filenames['baseline'])
    # human_rating = __read_human_rating(filenames['human_rating'])
    #
    #
    # logging.debug('.txt files read')
    #
    # return AnnotationData(sleep_profile, flow_events, arousals, baseline, human_rating)
    return None


def __read_sleep_profile(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:

        text_in_lines = f.readlines()
        header = {}
        df = pd.DataFrame()
        first_line_of_data = 0

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

        start_time = pd.Timestamp(header['Start Time'])
        rate = header['Rate']
        first_period = pd.Period(start_time, freq=rate)
        print(first_period + 1)
    f.close()
    return header, df





def __read_flow_events(filename: str):
    pass


def __read_arousals(filename: str):
    pass


def __read_baseline(filename: str):
    pass


def __read_human_rating(filename: str):
    pass


if __name__ == '__main__':
    pass
