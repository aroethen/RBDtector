import numpy as np
import pandas as pd

from typing import Dict, Any


class RawDataChannel:

    def __init__(self, signal_header: Dict[str, Any] = None, signal: np.ndarray = None):
        self._signal_header: Dict[str, Any] = signal_header
        self._signal: np.ndarray = signal

    def get_signal(self):
        return self._signal

    def set_signal(self, signal):
        self._signal = signal

    def get_signal_header(self):
        return self._signal_header

    def get_signal_as_pd_series(self):
        return pd.Series(self._signal)

    def get_sample_rate(self):
        return self._signal_header['sample_rate']
