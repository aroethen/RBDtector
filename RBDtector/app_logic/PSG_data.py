from input import input_reader as ir
from output import csv_writer


class PSGData:

    def __init__(self):
        self._input_path = ''
        self._output_path = ''

        self._raw_data = None          # content of edf file
        self._annotation_data = None   # content of txt files
        self._calculated_data = None

# PUBLIC FUNCTIONS
    def set_input_path(self, input_path):
        self._input_path = input_path

    def set_output_path(self, output_path):
        self._output_path = output_path

    def generate_output(self):
        self._read_data()
        self._calculated_data = self._process_data()
        human_rater_data = self._annotation_data.get_human_rater_data()
        csv_writer.write_output(self._calculated_data, human_rater_data)

# PRIVATE FUNCTIONS
    def _read_data(self):
        data = ir.read_input()
        self._raw_data = data[0]
        self._annotation_data = data[1]

    def _process_data(self):
        # TODO: returns calculated data in form that can be used by output writer
        pass
