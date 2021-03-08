#!/usr/bin/env python

# python modules
import logging

# internal modules
from gui import gui
from app_logic.PSG_data import PSGData

DEV = True
import os

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
        path = '/home/annika/WORK/RBDtector/Non-Coding-Content/EMG/EMGs'
        dirlist = os.listdir(path)
        for child in dirlist:
            abs_child = os.path.join(path, child)
            if os.path.isdir(abs_child):
                # if 'comparison_pickle' not in os.listdir(abs_child):
                data = PSGData(abs_child, abs_child)
                data.generate_output()

    else:
        gui.start_gui()
    # Final TODO: Catch all remaining errors, log them, show message with reference to logfile and exit with error code
