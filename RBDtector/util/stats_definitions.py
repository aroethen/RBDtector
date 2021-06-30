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

# #: All human rated labels
# QUALITIES = [
#     'LeftArmPhasic', 'LeftArmTonic', 'LeftArmAny',
#     'RightArmPhasic', 'RightArmTonic', 'RightArmAny',
#     'LeftLegPhasic', 'LeftLegTonic', 'LeftLegAny',
#     'RightLegPhasic', 'RightLegTonic', 'RightLegAny',
#     'ChinPhasic', 'ChinTonic', 'ChinAny',
#     'LeftArmArtifact', 'RightArmArtifact', 'LeftLegArtifact', 'RightLegArtifact', 'ChinArtifact'
# ]