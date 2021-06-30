# python modules
from typing import Tuple, Dict, List, Any

# external dependencies
import pandas as pd


class AnnotationData:
    """ AnnotationData is a storage class for all information that was read from input text files. It stores its data in
     the following format.

    :param sleep_profile:
        Contains all information of the sleep profile input file.
        (header: Dict, data: pd.DataFrame)
            header: header data of sleep profile input text file
            data:
                categorical column "sleep_phase" containing sleeping phase classification strings.
                index is a DatetimeIndex containing the timestamps from the sleep profile input text file

    :param flow_events:
        Contains all information of the flow events input file.
        (header: Dict, data: pd.DataFrame)
            header: header data of flow events input text file
            data:
                 Pandas DataFrame with default index containing the events per row described by the following columns:
                 'event_onset': onset time of event
                 'event_end_time': end time of event
                 'duration_in_seconds': event duration in seconds
                 'event': flow event classification as string

    :param arousals:
        Contains all information of the arousal classification input file.
        (header: Dict, data: pd.DataFrame)
            header: header data of arousal classification input text file
            data:
                 Pandas DataFrame with default index containing the events per row described by the following columns:
                 'event_onset': onset time of event
                 'event_end_time': end time of event
                 'duration_in_seconds': event duration in seconds
                 'event': arousal event classification as string

    :param baseline: (optional)
        Dictionary containing the info from the baseline text file.
            Keys: Baseline strings as coded in BASELINE_NAME
            Values: List of two timestamps, the first being the starting point of the baseline interval and the second
                    one its end point.

    :param human_rating: (optional)
        Contains all information of the human rating input file.
        (header: Dict, data: pd.DataFrame)
            header: header data of human rating input text file
            data:
                 Pandas DataFrame with default index containing the events per row described by the following columns:
                 'event_onset': onset time of event
                 'event_end_time': end time of event
                 'event': arousal event classification as string, which contains a combination of
                            HUMAN_RATING_LABEL and EVENT_TYPE
    """

    def __init__(self, sleep_profile, flow_events, arousals, baseline=None, human_rating=None):
        self.sleep_profile: Tuple[Dict[str, str], pd.DataFrame] = sleep_profile
        self.flow_events: Tuple[Dict[str, str], pd.DataFrame] = flow_events
        self.arousals: Tuple[Dict[str, str], pd.DataFrame] = arousals
        self.baseline: Dict[str, List[pd.Timestamp]] = baseline
        self.human_rating: List[Tuple[Dict[str, str]], pd.DataFrame] = human_rating
