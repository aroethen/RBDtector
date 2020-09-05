import os
import glob
from pyedflib import highlevel, edfreader


def readInput(directory_name):
    dir = os.path.abspath(directory_name)
    print(dir)
    edfs = glob.glob(dir + '*.edf')
    print(edfs)
    for edf in range(1):

        signals1, signal_headers1, header1 = highlevel.read_edf(dir)

        # for key, value in header1.items():
        #     print(key + ": ")
        #     print(value)
        #     print()
        for thing in signal_headers1:
            print(thing)

        for thingy in signals1:
            print(thingy)
            print(len(thingy))