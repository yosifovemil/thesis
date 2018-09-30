import itertools

from mscanpy import GlobalPheno, Stock, Accession

from framework.data.phenotype.plant import Plant
from framework.data.phenotype.pheno_reader.pheno_reader import PhenoReader
from framework.data.formats import *
import framework.util.user_details as user_details

class PhenoReaderMSCAN(PhenoReader):
	"""
	Reads data from MSCAN. 
	NOTE: The variables 'time_taken', 'who', 'pseudo_rep_no', 'leaf_ligule', 'stem_id', 
	'leaf_length_cm', 'leaf_width_cm' are returned as "", until Michael implements them into 
	mscanpy"""
	def __init__(self, location, **kwargs):
		self._location = location
		self._just_convert = 'just_convert' in kwargs.keys() and kwargs['just_convert']
		kwargs.pop('just_convert', None)

		self._query = self.perform_query(location, **kwargs)
		self._records = self.parse_result(self._query)

	def perform_query(self, location, **kwargs):
		user_details.init()

		query = GlobalPheno.get(user = user_details.get_username(),
								password = user_details.get_password(),
								field_plan = location.get_mscan_name(), **kwargs)
		return query
		
	def parse_result(self, query, include_species = False):
		#sort the rows by uid, date and rep no
		query_sorted = sorted(query, key = lambda x: (x.uid, x.pheno_date,
												x.get_pheno('pseudo_rep_no')))
	
		parsed = []
		if include_species:
			species = self.get_species(query_sorted)

		for entry in query_sorted:
			record = dict()
			keys = get_all_pheno_names()
			for key in keys:
				record[key] = self.get_value(entry, key)
			
			if include_species:
				record['species'] = species[entry.genotype]

			record = on_read(record, self._location, date_format = "%Y-%m-%dT00:00:00Z")
			parsed.append(record)

		return parsed

	def get_value(self, entry, key):
		if key == "EuroPheno.stock_id":
			return entry.uid
		else:
			return entry.get_pheno(key)

	def get_species(self, records):
		species = dict()
		genotypes = set([x.genotype for x in records])
		for genotype in genotypes:
			accession_name = genotype.split('#')[0]
			accession = Accession.get(user = user_details.get_username(), 
										password = user_details.get_password(),
										name = accession_name)[0]

			species[genotype] = accession.species

		return species
