from framework.data.location import Location
from framework.data.phenotype.pheno_reader.pheno_reader_csv import PhenoReaderCSV
from framework.data.phenotype.pheno_reader.yield_reader_csv import YieldReaderCSV
from framework.stats.bootstrap import Bootstrap, EmptyDataException
import framework.util.misc as misc
from framework.util.csv_io.csv_filewriter import CSVFileWriter

import framework.data.formats as formats
from scipy.stats import sem

from Queue import Queue

class BootstrapScenario:
	
	def __init__(self, bootstrap_n, f_location, run = True):
		years = [2011, 2012, 2014, 2015]
		locations = self.create_locations(years)
		self._data = self.attach_geno_data(self.read_BSBEC_data(locations))

		self._bootstrap_n = bootstrap_n
		self._f_location = f_location

		phenos = [formats.CANOPY_HEIGHT, formats.STEM_COUNT, formats.LEAF_WIDTH, 
				formats.LEAF_LENGTH, formats.DW_PLANT]

		if run:
			self.run_scenario(phenos)


	def run_scenario(self, phenos):
		plots = set([x[formats.UID]/10 for x in self._data])
		results = []
		queue = Queue()

		for plot in plots:
			print "plot - %d" % plot
			for pheno in phenos:
				print "%s" % pheno
				genotype = misc.assign_geno_bsbec(plot)
				self.test_se(plot, genotype, pheno, queue)

		while not queue.empty():
			results += queue.get()

		CSVFileWriter(self._f_location, results)

	def test_se(self, plot, genotype, pheno, queue):
		subset = [x for x in self._data if x[formats.UID]/10 == plot and x[pheno] != None]
		means = self.means_per_date(subset, pheno, plot)

		n_elements = 5
		se = []

		while n_elements <= len(means):
			try:
				result = self.get_se(n_elements, means, pheno)
			except EmptyDataException:
				n_elements += 1
				continue 

			se.append({'plot': plot,
						formats.GENOTYPE: genotype, 
						'n_elements': result['n_elements'],
						'pheno': pheno,
						'standard_error': result['se']})

			n_elements += 1

		queue.put(se)

	def get_se(self, n_elements, data, pheno):
		bootstrap = Bootstrap(data, pheno)

		subsets = bootstrap.generate_subsets(self._bootstrap_n, n_elements)
		se = []
		for subset in subsets:
			se.append(self.calc_se(subset, pheno))

		mean_se = sum(se)/len(se) #get the mean standard error of the mean
		return {'n_elements': bootstrap.get_n(),
				'se': mean_se} 

	def calc_se(self, subset, pheno):
		values = [x[pheno] for x in subset]
		return sem(values)

	def read_BSBEC_data(self, locations):
		pheno_data = self.bind_data(locations, PhenoReaderCSV)
		yield_data = self.bind_data(locations, YieldReaderCSV)
		return pheno_data + yield_data


	def bind_data(self, locations, reader_class):
		final_data = []
		for location in locations:
			final_data += reader_class(location).get_records()
		
		return final_data

	def attach_geno_data(self, data):
		for entry in data:
			uid = entry[formats.UID]
			entry['genotype'] = misc.assign_geno_bsbec(uid/10)
			entry['species'] = misc.get_species(uid/10)
		
		return data

	def create_locations(self, years):
		locations = []
		for year in years:
			locations.append(Location(Location.BSBEC, year))

		return locations

	def means_per_date(self, data, pheno, plot):
		dates = set([x[formats.DATE] for x in data])
		means = [] 
		for date in sorted(dates):
			total = 0
			day_subset = [x for x in data if x[formats.DATE] == date]
			for entry in day_subset:
				total += entry[pheno]

			total = total / len(day_subset)
			means.append({'plot': plot,
						formats.GENOTYPE: data[0][formats.GENOTYPE],
						pheno: total, 
						formats.DATE: date})

		return means
