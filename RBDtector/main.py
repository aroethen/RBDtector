#!/usr/bin/env python

# python modules
import logging

# internal modules
from gui import gui
from app_logic.PSG_data import PSGData

DEV = False

if __name__ == "__main__":
    logging.basicConfig(
        filename='logfile.txt',
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='### %d.%m.%Y %I:%M:%S ###  '
    )
    logging.info('\n----------- START -----------')
    logging.info('Starting GUI')

    if DEV:
        data = PSGData("/home/annika/WORK/RBDtector/TESTS_FILES/EMG_Test_02",
                       "/home/annika/WORK/RBDtector/TESTS_FILES/EMG_Test_02")
        data.generate_output()

    else:
        gui.start_gui()
    # Final TODO: Catch all remaining errors, log them, show message with reference to logfile and exit with error code
