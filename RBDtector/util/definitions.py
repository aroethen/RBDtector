
HUMAN_RATING_LABEL = {      #: Maps EMG channel name to EMG channel description in human rating labels
    'EMG': 'Chin',
    'PLM l': 'LeftLeg',
    'PLM r': 'RightLeg',
    'AUX': 'RightArm',
    'Akti.': 'LeftArm'
}

EVENT_TYPE = {      # Maps internal EMG event type to EMG event type description in human rating labels
    'tonic': 'Tonic',
    'intermediate': 'Any',
    'phasic': 'Phasic',
    'artifact': 'Artifact'
}

BASELINE_NAME = {       #: Maps EMG channel name to string identifier of the baseline inside the baseline input file
    'EMG': 'EMG REM baseline',
    'PLM l': 'PLMl REM baseline',
    'PLM r': 'PLMr REM baseline',
    'AUX': 'AUX REM baseline',
    'Akti.': 'AKTI REM baseline'
}

FILE_FINDER = {     #: Maps file type to its file name pattern used to find the file inside the input directory
    'edf': '*.edf',
    'sleep_profile': '*Sleep profile*',
    'flow_events': '*Flow Events*',
    'arousals': '*Classification Arousals*',
    'baseline': '*Start-Baseline*',
    'human_rating': '*Generic*'
}

SLEEP_CLASSIFIERS = {
    'artifact': 'A',
    'REM':  'REM'
}


def definitions_as_string():
    """ Convenience method to receive all definitions as formatted string. """
    return str(f"FILE_FINDER: {FILE_FINDER}\n"
               f"HUMAN_RATING_LABEL: {HUMAN_RATING_LABEL}\n"
               f"EVENT_TYPE: {EVENT_TYPE}\n"
               f"BASELINE_NAME: {BASELINE_NAME}\n"
               f"SLEEP_CLASSIFIERS: {SLEEP_CLASSIFIERS}\n"
               )
