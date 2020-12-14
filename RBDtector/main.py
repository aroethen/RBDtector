#!/usr/bin/env python

# python modules
import logging

# internal modules
from gui import gui
from app_logic.PSG_data import PSGData

DEV = True

def calculate_results(input_dir, output_dir):
    data = PSGData(input_dir, output_dir)
    data.generate_output()


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
        calculate_results("/home/annika/WORK/RBDtector/TESTS_FILES/iRBD0075",
                          "/home/annika/WORK/RBDtector/TESTS_FILES/iRBD0075")

    else:
        gui.start_gui()
    # Final TODO: Catch all remaining errors, log them, show message with reference to logfile and exit with error code
