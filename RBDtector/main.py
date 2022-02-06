#!/usr/bin/env python

# python modules
import configparser
import logging
import os
import sys
import traceback

# internal modules
from gui import gui
from app_logic.PSG_controller import PSGController, superdir_run
from util import settings

DEV = False

SUPERDIR = True


def read_config():
    config_exists = False
    try:
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(os.getcwd()), 'config.ini'))
        config_exists = True

        try:
            # Signal names of EMG channels in EDF files to be evaluated for RSWA
            config_signals_to_evaluate = config.get('Settings', 'SIGNALS_TO_EVALUATE', fallback=None)
            if config_signals_to_evaluate:
                settings.SIGNALS_TO_EVALUATE = [x.strip() for x in config_signals_to_evaluate.split(',')]

            # Artifact types to be excluded from evaluation
            settings.FLOW = config.getboolean('Settings', 'FLOW', fallback=settings.FLOW)
            settings.HUMAN_ARTIFACTS = config.getboolean('Settings', 'HUMAN_ARTIFACTS',
                                                         fallback=settings.HUMAN_ARTIFACTS)
            settings.SNORE = config.getboolean('Settings', 'SNORE', fallback=settings.SNORE)

            # Use manually defined static baselines from a baseline file instead of calculating adaptive baseline levels
            settings.HUMAN_BASELINE = config.getboolean('Settings', 'HUMAN_BASELINE',
                                                        fallback=settings.HUMAN_BASELINE)
        except configparser.NoSectionError as e:
            logging.info("Section 'Settings' not found in config file.")

    except EnvironmentError:
        config_exists = False
        logging.info('No config file found. Default settings are used.')


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
            path = '/media/SharedData/EMG/EMG-Scorings mGlu Nora'
            # path = '/media/SharedData/EMG/EMG-Scorings iRBD Nora'
            # path = '/home/annika/WORK/RBDtector/Non-Coding-Content/EMGs'
            path = 'D:/EMG/testifer'
            # path = '/localdata/EMG/EMG-Scorings iRBD Nora'

            superdir_run(path, dev_run=True)

        else:
            # path = '/media/SharedData/EMG/AUSLAGERUNG/iRBD0223'
            path = '/home/annika/WORK/RBDtector/Non-Coding-Content/EMGs/iRBD0216'
            _ = PSGController.run_rbd_detection(path, path)

    else:
        try:
            read_config()

            rbd_gui = gui.Gui()
            rbd_gui.mainloop()
        except BaseException as e:
            logging.error(f'Program terminated with unexpected error:\n {e}')
            logging.error(traceback.format_exc())
            print(f'An unexpected error occurred. Error message can be found in log file. Please contact the developer.')
            sys.exit(1)

    # Final TODO: Catch all remaining errors, log them, show message with reference to logfile and exit with error code
