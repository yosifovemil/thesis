from framework.util.csv_io.csv_filereader import CSVFileReader
from framework.data.phenotype.pheno_reader.pheno_reader import PhenoReader
import framework.data.formats as formats
from datetime import datetime
from framework.data.location import Location


class YieldReaderCSV(PhenoReader):
    """Reads the yield measurements from csv files"""

    def __init__(self, location, correct_records=True):
        self._location = location
        self._records = self.parse_records(location, correct_records)

    def parse_records(self, location, correct_records):
        content = CSVFileReader(location.get_harvest_dataf()).get_content()
        for entry in content:
            entry = formats.on_read(entry, self._location)

        # perform corrections and fixes
        if correct_records:
            for entry in content:
                # fix that annoying problem in BSBEC 2011 where they split the
                # harvest
                if location.get_name() == Location.BSBEC and \
                        entry[formats.DATE] == datetime(2011, 7, 19):
                    entry[formats.DATE] = datetime(2011, 7, 4)

                # fix entries without sub samples (2016 harvest, subsampling
                # was done only on the 12 plot harvest, as it is a more
                # accurate moisture measurement)
                if formats.DW_SUB not in entry.keys():
                    match = next(x for x in content if
                                 x[formats.DATE] == entry[formats.DATE] and
                                 x[formats.UID] == entry[formats.UID] and
                                 x[formats.PSEUDO_REP] == 0)

                    ratio = match[formats.DW_SUB] / match[formats.FW_SUB]
                    entry[formats.DW_PLANT] = entry[formats.FW_PLANT] * ratio

        return content
