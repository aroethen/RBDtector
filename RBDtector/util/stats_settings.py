
class StatsSettings:
    HUMAN_TONIC_GT50PCT = True

    @classmethod
    def to_string(cls):
        return str(f'HUMAN_TONIC_GT50PCT = {StatsSettings.HUMAN_TONIC_GT50PCT}\n')

