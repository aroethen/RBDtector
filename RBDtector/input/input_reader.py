# python modules
import os
import glob
from typing import Tuple, Dict
import logging

# third-party modules
from pyedflib import highlevel, edfreader

# internal modules
from app_logic.raw_data import RawData
from app_logic.raw_data_channel import RawDataChannel
from app_logic.annotation_data import AnnotationData


def read_input(directory_name: str) -> Tuple[RawData, AnnotationData]:
    # TODO: Read input into [RawPSGData, AnnotationData] and return it
    input_dir = os.path.abspath(directory_name)

    filenames = _find_files(input_dir)

    # print(input_dir)
    # edfs = glob.glob(input_dir + '*.edf')
    # print(edfs)
    # for edf in range(1):
    #
    #     signals1, signal_headers1, header1 = highlevel.read_edf(input_dir)
    #
    #     # for key, value in header1.items():
    #     #     print(key + ": ")
    #     #     print(value)
    #     #     print()
    #     for thing in signal_headers1:
    #         print(thing)
    #
    #     for thingy in signals1:
    #         print(thingy)
    #         print(len(thingy))


def _find_files(dir_path: str) -> Dict[str, str]:
    """ Finds EDF and text files in given directory and returns them as a dictionary
    with [edf, sleep_profile, flow_events, arousals, baseline, human_rating] as keys
    and the respective validated file names as values """
    pass
