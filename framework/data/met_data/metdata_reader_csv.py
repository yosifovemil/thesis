from datetime import datetime

from framework.util.csv_io.csv_filereader import CSVFileReader
from framework.data.met_data.metdata_reader import MetDataReader
import framework.data.formats as formats
from framework.data.met_data.degree_days import degree_days


class MetDataReaderCSV(MetDataReader):

    def __init__(self, location, t_base):
        self._met_data = CSVFileReader(location.get_met_data()).get_content()
        self._t_base = t_base

        for reading in self._met_data:
            reading[formats.DATE] = datetime.strptime(reading['Date'],
                                                      "%d/%m/%Y")
            reading.pop('Date')

            reading[formats.PAR] = self.parse_float(reading[formats.PAR])
            reading[formats.T_MAX] = self.parse_float(reading[formats.T_MAX])
            reading[formats.T_MIN] = self.parse_float(reading[formats.T_MIN])
            reading[formats.RAINFALL] = \
                self.parse_float(reading[formats.RAINFALL])

            reading[formats.DD] = degree_days(reading[formats.T_MAX],
                                              reading[formats.T_MIN],
                                              self._t_base)

            for key in reading:
                if (reading[key] == "NA"):
                    reading[key] = None

    def parse_float(self, reading):
        if (reading == "NA"):
            return None
        else:
            return float(reading)
