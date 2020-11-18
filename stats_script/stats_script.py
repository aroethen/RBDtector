import pandas as pd
import matplotlib as plt
import numpy as np
from typing import List, Tuple
import os
import re
import logging
import datetime

IN_PERCENT_OF_REM_DURATION = False

FILE_FINDER = {
    'edf': '.edf',
    'sleep_profile': 'Sleep profile',
    'flow_events': 'Flow Events',
    'arousals': 'Classification Arousals',
    'baseline': 'Start-Baseline',
    'human_rating': 'Generic'
}

QUALITIES = [
    'LeftArmPhasic', 'LeftArmTonic', 'LeftArmAny',
    'RightArmPhasic', 'RightArmTonic', 'RightArmAny',
    'LeftLegPhasic', 'LeftLegTonic', 'LeftLegAny',
    'RightLegPhasic', 'RightLegTonic', 'RightLegAny',
    'ChinPhasic', 'ChinTonic', 'ChinAny',
    'LeftArmArtifact', 'RightArmArtifact', 'LeftLegArtifact', 'RightLegArtifact', 'ChinArtifact'
]


def find_all_human_rated_directories(directory_name) -> List[Tuple[str, str]]:
    """
    :returns list of Tuples of all somnography directories containing human ratings and sleep profiles.
                Format: List[Tuple[directory name, direcory path
    """

    # find all subdirectories of given directory
    subdirectories = [(d.name, d.path) for d in os.scandir(os.path.abspath(directory_name))
                      if d.is_dir()]
    print(subdirectories)

    # remove subdirectories without human rater files from subdirectories
    for subdir in subdirectories:
        found_human_rater_file = False
        found_sleep_profile = False
        for file in os.scandir(subdir[1]):
            filename = file.name
            if FILE_FINDER['human_rating'] in filename:
                found_human_rater_file = True
            elif FILE_FINDER['sleep_profile'] in filename:
                found_sleep_profile = True

        if not found_human_rater_file or not found_sleep_profile:
            subdirectories.remove(subdir)

        found_human_rater_file = False
        found_sleep_profile = False

    print(subdirectories)

    return subdirectories


def read_files_into_dataframes(human_rater_files) -> List[pd.DataFrame]:
    pass


# def add_RBD_label_to_dataframe_list(all_hr_files_as_dataframes_list):
#     """Adds boolean value isRBD to list inside dataframes list"""
#     pass


def calculate_REM_sleep_duration(sleep_profile_filepath):
    """
    :param sleep_profile_filepath: file path to a sleep profile .txt file
    :returns: duration of REM sleep in seconds
    """
    rem_duration = 0

    with open(sleep_profile_filepath[0], 'r', encoding='utf-8') as f:
        text_in_lines = f.readlines()

        # find period rate in seconds and first line of data after header
        first_line_of_data = 0
        rate_in_seconds = 0
        for index, line in enumerate(text_in_lines):
            split_list = line.split(':')
            if split_list[0] == 'Rate':
                rate_in_seconds = int(re.findall('^[\d]*', split_list[1].strip())[0])
            if split_list[0].isdecimal():
                first_line_of_data = index
                break

        # count instances of REM in sleep_profile data
        number_of_rem_periods = 0
        for line in text_in_lines[first_line_of_data:]:
            split_list = line.split(';')
            if split_list[1].strip().upper() == 'REM':
                number_of_rem_periods += 1

        rem_duration = number_of_rem_periods * rate_in_seconds

    logging.info('Sleep_profile file: {}'.format(sleep_profile_filepath))
    logging.info('Sleep profile rate: {}s'.format(rate_in_seconds))
    logging.info('REM duration: {} (= rate * REM occurrences in sleep profile)'.format(rem_duration))

    return rem_duration


def collect_data_for_table_row_from_directory(dirtuple):
    # create return list and add subject
    row = [dirtuple[0]]

    # find total REM sleep duration in seconds and add to return list
    rem_duration = calculate_REM_sleep_duration([f.path for f in os.scandir(dirtuple[1])
                                                 if FILE_FINDER['sleep_profile'] in f.name])
    row.append(rem_duration)

    # find human_rating filepath
    human_rating = [f.path for f in os.scandir(dirtuple[1])
                    if FILE_FINDER['human_rating'] in f.name][0]

    # walk through human rating file and add the timedeltas of the respective scoring qualities
    # to a list of the same shape as QUALITIES
    with open(human_rating, 'r', encoding='utf-8') as f:
        text_in_lines = f.readlines()

        quality_timedeltas = [datetime.timedelta()] * len(QUALITIES)

        # find first line of data and start date
        start_date = None

        first_line_of_data = 0
        for index, line in enumerate(text_in_lines):
            split_list = line.split(':')

            if 'Start Time' in split_list[0]:
                start_date = datetime.datetime.strptime(split_list[1].strip(), '%d.%m.%Y %H').date()
                continue

            if split_list[0].isdecimal():
                first_line_of_data = index
                break

        # go through data lines and fill quality_timedeltas accordingly
        date_change_occurred = False

        for line in text_in_lines[first_line_of_data:]:

            # extract event, onset and end time from line
            times, _, event = line.partition(';')
            event = event.strip()
            times = times.split('-')
            onset_time = datetime.datetime.strptime(times[0], '%H:%M:%S,%f').time()
            end_time = datetime.datetime.strptime(times[1], '%H:%M:%S,%f').time()

            # create correct datetime for onset and end times
            onset_after_midnight = 0
            end_after_midnight = 0

            if datetime.time(0, 0, 0) <= onset_time < datetime.time(12, 0, 0):
                onset_after_midnight = 1
            if datetime.time(0, 0, 0) <= end_time < datetime.time(12, 0, 0):
                end_after_midnight = 1

            onset_datetime = datetime.datetime.combine(start_date, onset_time)\
                             + datetime.timedelta(days=onset_after_midnight)
            end_datetime = datetime.datetime.combine(start_date, end_time) \
                           + datetime.timedelta(days=end_after_midnight)

            # calculate event duration and add to the events' respective cell in quality_timedeltas list
            event_duration = end_datetime - onset_datetime
            quality_timedeltas[QUALITIES.index(event)] += event_duration

        # transform timedeltas in quality_timedeltas to percent of total REM duration
        if IN_PERCENT_OF_REM_DURATION:
            quality_timedeltas = [(tdelta.total_seconds() / rem_duration) * 100 for tdelta in quality_timedeltas]
        else:
            quality_timedeltas = [tdelta.total_seconds() for tdelta in quality_timedeltas]

        # print([str(pretty) for pretty in quality_timedeltas])
        logging.info('Times of qualities in subject {}:\n'
                     '{}'.format(dirtuple[0], dict(zip(QUALITIES, quality_timedeltas))))
        row.extend(quality_timedeltas)

    return row


def generate_descripive_statistics():
    human_rated_dirs = find_all_human_rated_directories('../Testfiles/Output')
    table = []

    for dirtuple in human_rated_dirs:
        row = collect_data_for_table_row_from_directory(dirtuple)
        table.append(row)

    columns = ['Subject', 'Total REM duration']
    columns.extend(QUALITIES)

    df = pd.DataFrame(table, columns=columns)
    print(df)
    output_filename = 'human_scoring_table'

    if IN_PERCENT_OF_REM_DURATION:
        output_filename += '_in_pct_of_REM.csv'
    else:
        output_filename += '_in_total_seconds.csv'

    df.to_csv(output_filename)


if __name__ == "__main__":
    logging.basicConfig(
        filename='logfile.txt',
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='### %d.%m.%Y %I:%M:%S ###  '
    )
    logging.info('\n----------- START -----------')

    generate_descripive_statistics()
