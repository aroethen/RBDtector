#!/usr/bin/env python

# python modules
import logging
import os
import sys
import traceback
from datetime import datetime

import pandas as pd

# internal modules
from gui import gui
from app_logic.PSG_controller import PSGController
from app_logic.PSG import PSG
from util.error_for_display import ErrorForDisplay
from util.settings import Settings

DEV = True

SUPERDIR = True

if __name__ == "__main__":
    logging.basicConfig(
        filename='logfile.txt',
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='### %d.%m.%Y %I:%M:%S ###  '
    )
    logging.info('\n----------- START -----------')
    logging.info('Starting GUI')

    if DEV:
        if SUPERDIR:
            path = '/media/SharedData/EMG/AUSLAGERUNG'
            # path = '/media/SharedData/EMG/EMG-Scorings iRBD Nora'
            dirlist = os.listdir(path)
            reading_problems = []
            df_out_combined = pd.DataFrame()
            df_channel_combinations_combined = pd.DataFrame()
            first = True

            for child in dirlist:
            # for abs_child in ['/media/SharedData/EMG/morePSG-Data/iRBD0065', '/media/SharedData/EMG/morePSG-Data/iRBD0067', '/media/SharedData/EMG/morePSG-Data/iRBD0113', '/media/SharedData/EMG/morePSG-Data/iRBD0216', '/media/SharedData/EMG/morePSG-Data/iRBD0223', '/media/SharedData/EMG/morePSG-Data/iRBD0268', '/media/SharedData/EMG/morePSG-Data/iRBD0273', '/media/SharedData/EMG/morePSG-Data/iRBD0310']:
                abs_child = os.path.join(path, child)
                if os.path.isdir(abs_child):
                    # if 'comparison_pickle' not in os.listdir(abs_child):
                    try:
                        df_out, df_channel_combinations = PSGController.run_rbd_detection(abs_child, abs_child)

                        if first:
                            df_out_combined = df_out.copy()
                            df_channel_combinations_combined = df_channel_combinations.copy()
                            first = False
                        else:
                            df_out_combined = pd.concat([df_out_combined, df_out], axis=1)
                            df_channel_combinations_combined = pd.concat([df_channel_combinations_combined, df_channel_combinations])

                    except (OSError, ErrorForDisplay) as e:
                        print(f'Expectable error in file {abs_child}:\n {e}')
                        logging.error(f'Expectable error in file {abs_child}:\n {e}')
                        logging.error(traceback.format_exc())
                        reading_problems.append(abs_child)
                        continue
                    except BaseException as e:
                        print(f'Unexpected error in file {abs_child}:\n {e}')
                        logging.error(f'Unexpected error in file {abs_child}:\n {e}')
                        logging.error(traceback.format_exc())
                        reading_problems.append(abs_child)
                        continue

            df_out_combined = df_out_combined\
                .reindex(['Signal', 'Global', 'EMG', 'PLM l', 'PLM r', 'AUX', 'Akti.'], level=0)
            df_out_combined.transpose()\
                .to_excel(os.path.join(path, f'RBDtector_combined_results_{datetime.now()}.xlsx'))

            df_channel_combinations_combined\
                .to_excel(os.path.join(path, f'Channel_combinations_combined_{datetime.now()}.xlsx'))

            if len(reading_problems) != 0:
                logging.error(f'These files could not be read: {reading_problems}')
                print(f'These files could not be read: {reading_problems}')
        else:
            path = '/media/SharedData/EMG/AUSLAGERUNG/iRBD0223'
            _ = PSGController.run_rbd_detection(path, path)

    else:
        try:
            gui.start_gui()
        except BaseException as e:
            logging.error(f'Programm terminated with unexpected error:\n {e}')
            logging.error(traceback.format_exc())
            print(f'An unexpected error occurred. Error message can be found in log file. Please contact the developer.')
            sys.exit(1)

    # Final TODO: Catch all remaining errors, log them, show message with reference to logfile and exit with error code
