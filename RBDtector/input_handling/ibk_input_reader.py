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
from datetime import date

from typing import Tuple, Dict, List
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
               read_baseline=True, read_edf=True, read_human_rating=True) -> Tuple[RawData, AnnotationData]:

    filenames = input_reader.find_files(directory_name, FILE_FINDER_IBK.copy())
    if read_edf:
        raw_data: RawData = read_edf(filenames['edf'], signals_to_load.copy())
        filenames.pop('edf')
    else:
        raw_data = None

    annotation_data: AnnotationData = __read_txt_files(filenames)
    return raw_data, annotation_data

def __read_txt_files(filenames) -> AnnotationData:
    pass





