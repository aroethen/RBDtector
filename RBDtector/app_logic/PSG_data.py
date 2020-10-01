# internal modules
from input import input_reader as ir
from output import csv_writer

# python modules
import logging
from typing import List


class PSGData:

    def __init__(self, input_path: str = '', output_path: str = ''):
        self._input_path = input_path
        self._output_path = output_path

        self._raw_data = None          # content of edf file
        self._annotation_data = None   # content of txt files
        self._calculated_data = None
        logging.debug('New PSGData Object created')

# PUBLIC FUNCTIONS
    @property
    def input_path(self):
        return self._input_path

    @input_path.setter
    def input_path(self, input_path):
        self._input_path = input_path

    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, output_path):
        self._output_path = output_path

    def generate_output(self):
        logging.debug('PSGData starting to generate output')
        self._read_data()
        # self._calculated_data = self._process_data()
        human_rater_data = self._annotation_data.get_human_rating()
        csv_writer.write_output(self._output_path, human_rating=human_rater_data)

# PRIVATE FUNCTIONS
    def _read_data(self):
        logging.debug('PSGData starting to read input')
        data = ir.read_input(self._input_path)
        self._raw_data = data[0]
        self._annotation_data = data[1]

    def _process_data(self) -> List[List]:
        # TODO: returns calculated data in form that can be used by output writer
        pass
