
class AnnotationData:

    def __init__(self, sleep_profile=None, flow_events=None, arousals=None, baseline=None, human_rating=None):
        self.sleep_profile = sleep_profile
        self.flow_events = flow_events
        self.arousals = arousals
        self.baseline = baseline
        self.human_rating = human_rating
