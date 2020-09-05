import unittest
from RBDtector.input import input_reader

import os


class TestInputReader(unittest.TestCase):

    def test_find_EDFs_in_directory(self):
        filepath1 = os.path.abspath("../Testfiles/Output/iRBD0095/iRBD0095_Unbekannt_(1).edf")
        edfs = input_reader.find_EDFs_in_directory("../Testfiles/Output/iRBD0095")
        edf1 = os.path.abspath(edfs[0])
        self.assertEqual(filepath1, edf1)


if __name__ == "__main__":
    unittest.main()
