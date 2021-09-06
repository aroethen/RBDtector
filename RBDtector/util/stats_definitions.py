"""
Definitions as used in stats script.
"""

#: Maps file name pattern to file type
FILE_FINDER = {
    'edf': '.edf',
    'sleep_profile': 'Sleep profile',
    'flow_events': 'Flow Events',
    'arousals': 'Classification Arousals',
    'baseline': 'Start-Baseline',
    'human_rating': 'Generic',
    'human_rating_2': 'Generic_NO'
}

#: Signal names to evaluate during stats calculations
SIGNALS_TO_EVALUATE = ['EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.']


def stats_definitions_as_string():
    """ Convenience method to receive all definitions as formatted string. """
    return str(f"FILE_FINDER: {FILE_FINDER}\n"
               f"SIGNALS_TO_EVALUATE: {SIGNALS_TO_EVALUATE}\n"
               )
