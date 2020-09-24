import numpy as np


class RawDataChannel:

    def __init__(self, signal_header=None, signal: np.ndarray = None):
        self._signal_header = signal_header
        self._signal = signal

        # TODO: Add attributes and useful methods for getting calculation data
