#!/usr/bin/env python

# python modules
import logging

# internal modules
from gui import gui
from app_logic.PSG_controller import PSGController
from app_logic.PSG import PSG
from util.error_for_display import ErrorForDisplay
from util.settings import Settings

DEV = True
import os

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
        path = '/media/SharedData/EMG/testifer'
        # path = '/home/annika/WORK/RBDtector/Non-Coding-Content/Testfiles/test_artifact_menge'
        dirlist = os.listdir(path)
        reading_problems = []
        for child in dirlist:
            abs_child = os.path.join(path, child)
            if os.path.isdir(abs_child):
                # if 'comparison_pickle' not in os.listdir(abs_child):
                try:
                    # data = PSG(abs_child, abs_child)

                    # if Settings.DEV_READ_PICKLE_INSTEAD_OF_EDF:
                    #     data.use_pickled_df_as_calculated_data(os.path.join(abs_child, 'pickledDF'))
                    # else:
                    #     data.read_input()
                    #     data.detect_RBD_arousals()
                    #
                    # data.write_results()
                    PSGController.run_rbd_detection(abs_child, abs_child)
                    # data.generate_output()


                except (OSError, ErrorForDisplay) as e:
                    print(e)
                    reading_problems.append(abs_child)
                    continue

        if len(reading_problems) != 0:
            print(f'These files could not be read: {reading_problems}')

        # path = '/media/SharedData/EMG/testifer/test1'
        # PSGController.run_rbd_detection(path, path)
        # data = PSG(path, path)
        # data.generate_output()

    else:
        gui.gui.start_gui()
    # Final TODO: Catch all remaining errors, log them, show message with reference to logfile and exit with error code
