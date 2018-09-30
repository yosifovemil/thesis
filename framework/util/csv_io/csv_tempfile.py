import tempfile
import os

from framework.util.csv_io.csv_filewriter import CSVFileWriter


class CSVTempFile:
    def __init__(self, data):
        self._f = tempfile.NamedTemporaryFile()
        CSVFileWriter(self._f.name, data)

    def get_name(self):
        return self._f.name

    def close(self):
        self._f.close()
