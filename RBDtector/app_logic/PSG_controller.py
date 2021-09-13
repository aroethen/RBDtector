import logging
import os

from util.settings import Settings
from app_logic.PSG import PSG


class PSGController:
    """High-level controller for PSG evaluation functionality. Provides easy to use functions for RBD event detection."""

    @staticmethod
    def run_rbd_detection(input_path: str, output_path: str):
        data = PSG(input_path, output_path)

        if Settings.DEV_READ_PICKLE_INSTEAD_OF_EDF:
            data.use_pickled_df_as_calculated_data(os.path.join(input_path, 'pickledDF'))
        else:
            data.read_input(Settings.SIGNALS_TO_EVALUATE)
            data.detect_RBD_arousals()

        data.write_results()

