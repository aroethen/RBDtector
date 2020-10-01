import pandas as pd
import numpy as np
import os

# class csv_writer():
#     def __init__(self):
#         pass


def write_output(output_path, calculated_data=None, human_rating=None):
    df = human_rating[1]
    onset = pd.to_datetime(df['event_onset'], unit='ms').dt.time
    onset = onset.astype(str)
    onset = onset.apply(lambda x: x[:-3])
    duration_in_seconds = (df['event_end_time'] - df['event_onset']) / np.timedelta64(1, 's')
    df_for_edfBrowser = pd.DataFrame(pd.concat([onset, duration_in_seconds, df['event']], axis=1))
    df_for_edfBrowser.columns = ['event_onset', 'duration', 'event']
    print(onset)
    df_for_edfBrowser.to_csv(os.path.join(output_path, 'csv_annotations_for_edfBrowser.txt'), index=False)

    df.to_csv(os.path.join(output_path, 'csv_output.txt'), index=False)


def __cut_off_nanoseconds(x):
    return x[:-3]
