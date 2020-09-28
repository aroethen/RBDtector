import numpy as np
from typing import Dict, Any


class RawDataChannel:

    def __init__(self, signal_header: Dict[str, Any] = None, signal: np.ndarray = None):
        self._signal_header: Dict[str, Any] = signal_header
        self._signal: np.ndarray = signal

        # TODO: Add attributes and useful methods for getting calculation data
