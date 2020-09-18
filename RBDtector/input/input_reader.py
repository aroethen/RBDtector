# python modules
import os
import glob
from typing import Tuple, Dict, List
import logging

# third-party modules
from pyedflib import highlevel, edfreader

# internal modules
from app_logic.raw_data import RawData
from app_logic.raw_data_channel import RawDataChannel
from app_logic.annotation_data import AnnotationData


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

    filenames = _find_files(directory_name)
    raw_data: RawData = read_edfs(filenames['edf'])
    annotation_data: AnnotationData = read_txt_files(filenames)

    return raw_data, annotation_data


def _find_files(directory_name: str) -> Dict[str, str]:
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
    except OSError:
        logging.exception()
        raise OSError('Directory "{}" does not exist'.format(directory_name))
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


def read_edfs(edf: str) -> RawData:

    signals, signal_headers, header = highlevel.read_edf(edf)
    print(type(header))
    print(type(signal_headers))
    print(type(signals))
    for key, value in header.items():
        print(key + ": ")
        print(value)
        print()

    for thing in signal_headers:
        print(thing)
    #
    # for thingy in signals:
    #     print(thingy)
    #     print(len(thingy))
    pass


def read_txt_files(files: Dict[str, str]) -> AnnotationData:
    pass
