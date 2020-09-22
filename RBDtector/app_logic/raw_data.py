# python modules
from typing import Dict

# internal modules
from app_logic.raw_data_channel import RawDataChannel


class RawData:

    def __init__(self, header: Dict[str, str] = None):
        """ Constructor of class RawData.
        :param header: header of .edf file as read by pyedflib.highlevel (Dict[str, str or List[List]]).
        Expected keys: 'startdate', 'patientcode', 'birthdate', 'admincode', 'gender', 'epuipment', 'annotations'
        """
        self._header = header
        self._eeg_data: Dict[str, RawDataChannel] = None
        self._eog_data: Dict[str, RawDataChannel] = None
        self._emg_data: Dict[str, RawDataChannel] = None
        self._ecg_data: Dict[str, RawDataChannel] = None
        self._O2_data: Dict[str, RawDataChannel] = None
