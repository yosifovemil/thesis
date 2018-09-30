# This class is a way to use static dates for data without emergence
# measurements


class FLLReaderStatic:
    def __init__(self, location, date):
        self._location = location
        self._date = date

    def get_plot_fll(self, plot_n):
        return self._date

    def get_genotype_fll(self, genotype):
        return self._date
