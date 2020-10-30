# python modules
from typing import Tuple, Dict

# external dependencies
import pandas as pd


class AnnotationData:

    def __init__(self, sleep_profile=None, flow_events=None, arousals=None, baseline=None, human_rating=None):
        self.sleep_profile: Tuple[Dict[str, str], pd.DataFrame] = sleep_profile
        self.flow_events: Tuple[Dict[str, str], pd.DataFrame] = flow_events
        self.arousals: Tuple[Dict[str, str], pd.DataFrame] = arousals
        self.baseline = baseline
        self.human_rating: Tuple[Dict[str, str], pd.DataFrame] = human_rating

    def get_human_rating(self):
        return self.human_rating
