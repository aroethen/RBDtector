import unittest
import RBDtector.input_handling
from input_handling import input_reader


import os



class TestInputReader(unittest.TestCase):

    def test_read_input(self):
        filepath1 = os.path.abspath("../Testfiles/Output/iRBD0095/iRBD0095_Unbekannt_(1).edf")
        edfs = RBDtector.input_handling.input_reader.__find_files("../Testfiles/Output/iRBD0095")
        edf1 = os.path.abspath(edfs[0])
        self.assertEqual(filepath1, edf1)


if __name__ == "__main__":
    unittest.main()
