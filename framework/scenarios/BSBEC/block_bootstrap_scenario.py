from framework.data.location import Location
from framework.data.phenotype.pheno_reader.pheno_reader_csv import PhenoReaderCSV
from framework.data.phenotype.pheno_reader.yield_reader_csv import YieldReaderCSV
from framework.stats.bootstrap import Bootstrap, EmptyDataException
from framework.util.misc import *
from framework.util.csv_io.csv_filewriter import CSVFileWriter

import framework.data.formats as formats
import math
import time
import os
import copy

from threading import Thread
from Queue import Queue

import numpy as np
import scipy.stats as stats
import random

class BlockBootstrapScenario:
	
	def __init__(self, block_length, bootstrap_n, f_location):
		years = [2011, 2012, 2014, 2015]
		locations = self.create_locations(years)

		self._f_location = f_location
		self._data = self.attach_geno_data(self.read_BSBEC_data(locations))

		phenos = [formats.CANOPY_HEIGHT, formats.STEM_COUNT, formats.LEAF_WIDTH, 
				formats.LEAF_LENGTH, formats.DW_PLANT]

		self.run(phenos, block_length, bootstrap_n)


	def run(self, phenos, block_length, bootstrap_n):
		plots = set([x['plot'] for x in self._data])
		result = [] 
		for plot in plots:
			print "Plot %d" % plot
			for pheno in phenos:
				print "Pheno %s" % pheno
				all_measurements = [x for x in self._data if x['plot'] == plot and x[pheno] != None]

				#take means per date
				means = self.means_per_date(all_measurements, pheno, plot)
				
				#detrend the data
				detrended = self.detrend_data(means, pheno)
	
				#create the bootstrap blocks
				blocks = []
				for i in range(len(detrended) - block_length + 1):
					block = []
					for j in range(block_length):
						block.append(detrended[i + j])
					
					blocks.append(block)

				#run bootstraps with different number of blocks
				for n in range(5, len(detrended) - block_length + 1):
					std_errs = []
					#generate n number of bootstraps
					for i in range(bootstrap_n):
						bootstrap = []
						for j in range(n):
							bootstrap += random.choice(blocks)

						std_errs.append(stats.sem([x[pheno] for x in bootstrap]))

					std_err = sum(std_errs)/len(std_errs)

					result.append({'plot': plot, 
									'pheno': pheno, 
									'n_blocks': n,
									'standard_error': std_err, 
									formats.GENOTYPE: detrended[0][formats.GENOTYPE]})


		CSVFileWriter(self._f_location, result)
				

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

	def detrend_data(self, data, pheno):
		data.sort(key = lambda x: x[formats.DATE])
		detrended = []
		for entry in data:
			if len(detrended) == 0:
				new_entry = copy.deepcopy(entry)
				new_entry[pheno] = 0
				detrended.append(new_entry)
			else:
				#compare years
				p_index = data.index(entry) - 1
				if entry[formats.DATE].strftime("%Y") != data[p_index][formats.DATE].strftime("%Y"):
					#a new year means 0 change
					diff = 0
				else:
					diff = entry[pheno] - data[p_index][pheno]

				new_entry = copy.deepcopy(entry)
				new_entry[pheno] = diff
				detrended.append(new_entry)

		return detrended
			

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
			entry['genotype'] = assign_geno_bsbec(uid/10)
			entry['species'] = get_species(uid/10)
			entry['plot'] = uid/10
		
		return data

	def create_locations(self, years):
		locations = []
		for year in years:
			locations.append(Location(Location.BSBEC, year))

		return locations

