from framework.data.location import Location
from framework.data.phenotype.sunscan_reader.sunscan_reader import SunScanReader
from framework.data.phenotype.pheno_reader.pheno_reader_csv import PhenoReaderCSV
from framework.data.phenotype.pheno_reader.yield_reader_csv import YieldReaderCSV
from framework.data import formats
import framework.util.misc as misc

class SunScanVsPhenoScenario:

	_years = [2011, 2012, 2014, 2015]
	_reader_classes = PhenoReaderCSV, YieldReaderCSV, SunScanReader
	_location_name = Location.BSBEC

	def __init__(self):
		self._data = self._read_data()
		self.pair_measurements(self._data)
		

	def pair_measurements(self, data):
		final_data = []
		for year in self._years:
			year_data = [x for x in data if x[formats.DATE].year == year]
			uids = set([x[formats.UID] for x in year_data])

			#sunscan measurements are done on the whole plot, not individual plants
			#filter the uids to get plant uids
			plant_uids = sorted([x for x in uids if x % 10 > 0])
			for uid in plant_uids:
				plant_data = [x for x in year_data if x[formats.UID] == uid and
														formats.TRANSMISSION not in x.keys()]

				leaf_areas = misc.calculate_leaf_areas(plant_data)
				
				#append other pheno measurements 
				if year < 2013:
					other_phenos = [formats.TRANSMISSION, formats.CANOPY_HEIGHT, formats.STEM_COUNT]
					plot_uid = uid - (uid % 10)
					other_data = [x for x in year_data if x[formats.UID] == plot_uid]
					matching = 1
				else:
					other_phenos = [formats.TRANSMISSION]
					other_data = [x for x in year_data if x[formats.UID] == uid and 
															formats.TRANSMISSION in x.keys()]
					matching = 2

				for entry in leaf_areas:
					for pheno in other_phenos:
						entry[pheno] = self._get_matching_phenos(entry[formats.DATE],
																pheno,
																other_data,
																matching)


						#DEBUG
						if entry[pheno] == None:
							print "Missing %s" % pheno
							from pprint import pprint
							pprint(entry)
							print "\n"

				final_data += leaf_areas
	

	def _read_data(self):
		data = []
		for reader_class in self._reader_classes:
			for year in self._years:
				location = Location(self._location_name, year)
				data += reader_class(location).get_records()

		#drop the useless pheno traits
		data = formats.drop_none_phenos(data)

		return data


	def _get_matching_phenos(self, date, pheno, plot_data, matching):
		data = [x for x in plot_data if pheno in x.keys()]
		ranking = []

		for entry in data:
			diff = entry[formats.DATE] - date
			ranking.append({'diff': diff, 'entry': entry})

		ranking.sort(key = lambda x: abs(x['diff'].total_seconds()))
		total = 0 

		for i in range(matching):
			if ranking[i]['diff'].days < 8:
				total += ranking[i]['entry'][pheno]

		total = total / matching 
	
		if total == 0:
			return None

		else:
			return total
