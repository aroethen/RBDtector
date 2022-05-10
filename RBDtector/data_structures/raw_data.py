# python modules
from typing import Dict

# internal modules
from data_structures.raw_data_channel import RawDataChannel


class RawData:

    def __init__(self, header: Dict[str, str] = None, data_channels: Dict[str, RawDataChannel] = None):
        """ Constructor of class RawData.

        :param header: header of .edf file as read by pyedflib.highlevel (Dict[str, str or List[List]]).
                        Necessarily expected keys: 'startdate'
        :param data_channels:
        """
        self._header: Dict[str, str] = header
        self._data_channels: Dict[str, RawDataChannel] = data_channels

    def set_data_channels(self, data_channels: Dict[str, RawDataChannel]):
        self._data_channels = data_channels

    def get_data_channels(self) -> Dict[str, RawDataChannel]:
        return self._data_channels

    def set_header(self, header: Dict[str, str]):
        self._header = header

    def get_header(self) -> Dict[str, str]:
        return self._header

    def add_channel(self, name: str, channel: RawDataChannel):
        if name in self._data_channels.keys():
            raise KeyError('Key "' + name + '" already exists in _data_channels.')
        else:
            self._data_channels[name]: RawDataChannel

    @property
    def header(self):
        return self._header
