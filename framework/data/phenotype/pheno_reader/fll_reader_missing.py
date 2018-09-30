# This class fixes the missing emergence data problem for BSBEC

from framework.data.location import Location
from framework.data.phenotype.pheno_reader.fll_reader import FLLReader

from datetime import datetime


class FLLReaderMissing:
    def __init__(self, location):
        if location.get_name() == Location.BSBEC:
            self._location = location
        else:
            raise Exception("Unimplemented for %s" % location.get_name())

    def _fix_start_date(self, years, genotype):
        start_dates = []
        for year in years:
            location = Location(self._location.get_name(), year)
            fll_reader = FLLReader(location)
            fll_date = fll_reader.get_genotype_fll(genotype)
            start_dates.append(int(fll_date.strftime("%j")))

        jul = sum(start_dates)/len(start_dates)
        date_string = '%d %s' % (jul, self._location.get_year())
        start_date = datetime.strptime(date_string, "%j %Y")
        return start_date

    def get_genotype_fll(self, genotype):
        return self._fix_start_date([2011, 2012],
                                    genotype)

    def get_plot_fll(self, plot_n):
        raise Exception("Unimplemented function!")
