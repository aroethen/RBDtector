

# Internally used sample rate
RATE = 256

# Signal names of EMG channels in EDF files to be evaluated for RSWA
SIGNALS_TO_EVALUATE = ['EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.']

# Artifact types to be excluded from evaluation
FLOW = False
HUMAN_ARTIFACTS = False
SNORE = True

# Use manually defined static baselines from a baseline file instead of calculating adaptive baseline levels
HUMAN_BASELINE = False

# Internal calculation variables - change default values only in case of optimization study
COUNT_BASED_ACTIVITY = False
MIN_SUSTAINED = 0.1
MAX_GAP_SIZE = 0.25
CHUNK_SIZE = '30L'
WITH_OFFSET = True
OFFSET_SIZE = '15L'

# Development settings
DEV = False


def settings_as_string():
    return str(f'DEV = {DEV}\n' +
               f'RATE = {RATE}\n' +
               f'FLOW = {FLOW}\n' +
               f'HUMAN_ARTIFACTS = {HUMAN_ARTIFACTS}\n' +
               f'HUMAN_BASELINE = {HUMAN_BASELINE}\n' +
               f'SNORE = {SNORE}\n' +
               f'COUNT_BASED_ACTIVITY = {COUNT_BASED_ACTIVITY}\n' +
               f'MIN_SUSTAINED = {MIN_SUSTAINED}\n' +
               f'MAX_GAP_SIZE = {MAX_GAP_SIZE}\n' +
               f'CHUNK_SIZE = {CHUNK_SIZE}\n' +
               f'WITH_OFFSET = {WITH_OFFSET}\n' +
               f'OFFSET_SIZE = {OFFSET_SIZE}\n' +
               f'SIGNALS_TO_EVALUATE = {str(SIGNALS_TO_EVALUATE)}\n'
               )
