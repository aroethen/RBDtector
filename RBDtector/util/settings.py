
class Settings:
    DEV = False
    SPLINES = True
    RATE = 256
    FREQ = '3.90625ms'
    FLOW = False
    HUMAN_ARTIFACTS = True
    COUNT_BASED_ACTIVITY = False
    MIN_SUSTAINED = 0.1
    MAX_GAP_SIZE = 0.25
    CHUNK_SIZE = '30L'
    WITH_OFFSET = True
    OFFSET_SIZE = '15L'
    LOW_PASS = 12
    VERBOSE = True
    SHOW_PLOT = False
    SIGNALS_TO_EVALUATE = ['EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.']

    @classmethod
    def to_string(cls):
        return str(f'DEV = {Settings.DEV}\n' + \
                f'RATE = {Settings.RATE}\n' + \
                f'FREQ = {Settings.FREQ}\n' + \
                f'FLOW = {Settings.FLOW}\n' + \
                f'HUMAN_ARTIFACTS = {Settings.HUMAN_ARTIFACTS}\n' + \
                f'COUNT_BASED_ACTIVITY = {Settings.COUNT_BASED_ACTIVITY}\n' + \
                f'MIN_SUSTAINED = {Settings.MIN_SUSTAINED}\n' + \
                f'MAX_GAP_SIZE = {Settings.MAX_GAP_SIZE}\n' + \
                f'CHUNK_SIZE = {Settings.CHUNK_SIZE}\n' + \
                f'WITH_OFFSET = {Settings.WITH_OFFSET}\n' + \
                f'OFFSET_SIZE = {Settings.OFFSET_SIZE}\n' + \
                f'LOW_PASS = {Settings.LOW_PASS}\n' + \
                f'SPLINES = {Settings.SPLINES}\n' + \
                f'VERBOSE = {Settings.VERBOSE}\n' + \
                f'SHOW_PLOT = {Settings.SHOW_PLOT}\n' + \
                f'SIGNALS_TO_EVALUATE = {str(Settings.SIGNALS_TO_EVALUATE)}\n')
