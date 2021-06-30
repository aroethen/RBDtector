
#: Maps EMG channel name to EMG channel description in human rating labels
HUMAN_RATING_LABEL = {
    'EMG': 'Chin',
    'PLM l': 'LeftLeg',
    'PLM r': 'RightLeg',
    'AUX': 'RightArm',
    'Akti.': 'LeftArm'
}

# Maps internal EMG event type to EMG event type description in human rating labels
EVENT_TYPE = {
    'tonic': 'Tonic',
    'intermediate': 'Any',
    'phasic': 'Phasic',
    'artefact': 'Artifact'
}

#: Maps EMG channel name to string identifier of the baseline inside the baseline input file
BASELINE_NAME = {
    'EMG': 'EMG REM baseline',
    'PLM l': 'PLMl REM baseline',
    'PLM r': 'PLMr REM baseline',
    'AUX': 'AUX REM baseline',
    'Akti.': 'AKTI REM baseline'
}

