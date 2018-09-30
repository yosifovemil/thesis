from framework.data.phenotype.pheno_reader.pheno_reader import PhenoReader
from framework.util.csv_io.csv_filereader import CSVFileReader

import framework.data.formats as formats


class PhenoReaderCSV(PhenoReader):

    _CD_NON_EXISTANT = -100.0
    _CD_DEAD = -10.0
    _CD_MISSING = -1000.0

    def __init__(self, location):
        self._location = location
        self._filename = self._location.get_pheno_dataf()
        self._records = self.parse_file(self._filename)

    def parse_file(self, filename):
        content = CSVFileReader(filename).get_content()
        content = self.parse_content(content)
        return content

    def parse_content(self, content):
        final_content = []
        for entry in content:
            entry = formats.on_read(entry, self._location)

            # Handle Chris Davey's masking and dead values (if any of them are
            # left)
            for key in entry.keys():
                if entry[key] in [self._CD_DEAD, self._CD_MISSING,
                                  self._CD_NON_EXISTANT]:
                    import ipdb
                    ipdb.set_trace()

            try:
                self.sanity_check(entry)
            except AssertionError as e:
                from pprint import pprint
                pprint(e)
                pprint(entry)
                import ipdb
                ipdb.set_trace()
                print("BOOM")

            final_content.append(entry)

        return final_content

    def sanity_check(self, entry):
        if entry[formats.PSEUDO_REP] > 0:
            assert(entry[formats.MEASUREMENT] >= 0)
            if entry[formats.MEASUREMENT] == 0:

                if formats.FLOWERING_SCORE in entry.keys():
                    assert(entry[formats.FLOWERING_SCORE] in range(6))

                if formats.STEM_ID in entry.keys():
                    assert(entry[formats.STEM_ID] > 0)

                if formats.STEM_COUNT in entry.keys():
                    assert((entry[formats.STEM_COUNT] <= 500.0 and
                           entry[formats.STEM_COUNT] >= 0.0))

                if formats.CANOPY_HEIGHT in entry.keys():
                    assert((entry[formats.CANOPY_HEIGHT] < 370.0 and
                           entry[formats.CANOPY_HEIGHT] >= 0.0))

            else:
                assert(entry[formats.LEAF_LENGTH] < 125.0 and
                       entry[formats.LEAF_LENGTH] >= 0.0)
                assert(entry[formats.LEAF_WIDTH] < 5.0 and
                       entry[formats.LEAF_WIDTH] >= 0.0)

    def masking_allowed(self):
        return int(self._location._year) < 2013
