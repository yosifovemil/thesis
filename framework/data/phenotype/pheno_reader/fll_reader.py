from framework.util.csv_io.csv_filereader import CSVFileReader
from datetime import datetime


class FLLReader:
    def __init__(self, location):
        self._location = location
        self.load_content()

    def load_content(self):
        fll_f = self._location.get_fll_location()
        reader = CSVFileReader(fll_f)
        content = reader.get_content()
        for entry in content:
            entry['Date'] = datetime.strptime("%s %s" %
                                              (entry['fll_day'],
                                               entry['year']), "%j %Y")

        self._content = content

    def get_plot_fll(self, plot_n):
        plot_n = str(plot_n)  # just in case it is not
        return next(entry['Date'] for entry in self._content
                    if entry['Plot'] == plot_n)

    def get_genotype_fll(self, genotype):
        days = [float(entry['Date'].strftime("%j")) for entry in self._content
                if entry['Genotype'] == genotype]

        avg_day = sum(days)/len(days)
        avg_date = datetime.strptime("%d %d" % (int(avg_day),
                                     self._location.get_year()), "%j %Y")
        return avg_date
