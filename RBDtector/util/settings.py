
class Settings:
    # Dev
    DEV = False
    DEV_READ_PICKLE_INSTEAD_OF_EDF = False

    # Internally used sample rate
    RATE = 256

    # Signal names of EMG channels in EDF files to be evaluated for RSWA
    SIGNALS_TO_EVALUATE = ['EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.']

    # Artifact types to be excluded from evaluation
    FLOW = False
    HUMAN_ARTIFACTS = False
    SNORE = False
    EX = False

    # Use manually defined static baselines from a baseline file instead of calculating adaptive baseline levels
    HUMAN_BASELINE = False

    # Internal calculation variables - change default values only in case of optimization study
    COUNT_BASED_ACTIVITY = False
    MIN_SUSTAINED = 0.1
    MAX_GAP_SIZE = 0.25
    CHUNK_SIZE = '30L'
    WITH_OFFSET = True
    OFFSET_SIZE = '15L'


    @classmethod
    def to_string(cls):
        return str(f'DEV = {Settings.DEV}\n' +
                   f'DEV_READ_PICKLE_INSTEAD_OF_EDF = {Settings.DEV_READ_PICKLE_INSTEAD_OF_EDF}\n' +
                   f'RATE = {Settings.RATE}\n' +
                   f'FLOW = {Settings.FLOW}\n' +
                   f'HUMAN_ARTIFACTS = {Settings.HUMAN_ARTIFACTS}\n' +
                   f'HUMAN_BASELINE = {Settings.HUMAN_BASELINE}\n' +
                   f'SNORE = {Settings.SNORE}\n' +
                   f'EX = {Settings.EX}\n' +
                   f'COUNT_BASED_ACTIVITY = {Settings.COUNT_BASED_ACTIVITY}\n' +
                   f'MIN_SUSTAINED = {Settings.MIN_SUSTAINED}\n' +
                   f'MAX_GAP_SIZE = {Settings.MAX_GAP_SIZE}\n' +
                   f'CHUNK_SIZE = {Settings.CHUNK_SIZE}\n' +
                   f'WITH_OFFSET = {Settings.WITH_OFFSET}\n' +
                   f'OFFSET_SIZE = {Settings.OFFSET_SIZE}\n' +
                   f'SIGNALS_TO_EVALUATE = {str(Settings.SIGNALS_TO_EVALUATE)}\n'
                   )
