#!/usr/bin/env python

# python modules
import logging

# internal modules
from gui import gui
from app_logic.PSG_data import PSGData


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
    gui.start_gui()
