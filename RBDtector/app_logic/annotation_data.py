
class AnnotationData:

    def __init__(self, sleep_profile=None, flow_events=None, arousals=None, baseline=None, human_rating=None):
        self._sleep_profile = sleep_profile
        self._flow_events = flow_events
        self._arousals = arousals
        self._baseline = baseline
        self._human_rating = human_rating