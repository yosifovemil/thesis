from framework.data.location import Location
from framework.data.phenotype.pheno_reader.pheno_reader_csv import PhenoReaderCSV
from framework.data.phenotype.pheno_reader.yield_reader_csv import YieldReaderCSV
from framework.data.phenotype.pheno_reader.pheno_reader_mscan import PhenoReaderMSCAN
from framework.data.phenotype.pheno_reader.fll_reader import FLLReader
from framework.models.two_stage_lm import TwoStageLM

import framework.data.formats as formats
from framework.util.misc import *

from scipy.interpolate import interp1d
from pprint import pprint
from datetime import datetime
from datetime import timedelta
import copy
import time

from framework.util.csv_io.csv_filewriter import CSVFileWriter

class GapFill:
	"""
	This is the gap filling scenario class - uses JKI 15 2nd year data to gap fill yield data 
	in the BSBEC dataset"""

	PHENO_VARS = [formats.STEM_COUNT, formats.CANOPY_HEIGHT, formats.LEAF_AREA]

	def __init__(self):
		#self._locations = [Location(Location.BSBEC, 2011), Location(Location.BSBEC, 2012)]
		self._locations = [Location(Location.BSBEC,2011)]
		self._jki_data = self.get_training_data()

		#BSBEC data handling
		self._bsbec_data = self.get_BSBEC_data(self._locations)
		injected_bsbec_data = self.inject_records(self._bsbec_data)

		self._interpolated = self.interpolate(injected_bsbec_data)
		self._interpolated = self.attach_geno_data(self._interpolated)
		self._yield_data = self.attach_geno_data(self._bsbec_data[formats.DW_PLANT])

		self.model = TwoStageLM(self._jki_data, self._interpolated, self._yield_data)
		self.model.gap_fill_data("EMI-11", "sinensis")
		self.model.gap_fill_data("Sac-5", "sacchariflorus")
		self.model.gap_fill_data("Goliath", ['sacc/sin', 'hybrid'])
		self.model.gap_fill_data("Giganteus", ['sacc/sin', 'hybrid'])
		CSVFileWriter("/home/eey9/Scratch/ann/data.csv", self.model._final_data)
		CSVFileWriter("/home/eey9/Scratch/ann/yield.csv", self._yield_data)

	### PROCESS JKI DATA ###

	def get_training_data(self):
		"""
		Grab the EuroPheno.stock_id, canopy_height, stem_count, species and dry_weight measurements
		from the loaded JKI data"""
		jki_data = self.get_JKI_data()

		dropped = 0
		final_data = []
		uids = set([x[formats.UID] for x in jki_data])
		for uid in uids:
			record = dict()
			entries = [x for x in jki_data if x[formats.UID] == uid]
			entries.sort(key = lambda x: x[formats.DATE]) #just making sure
			if len(entries) == 2:
				record[formats.UID] = uid
				record[formats.CANOPY_HEIGHT] = entries[0][formats.CANOPY_HEIGHT]
				record[formats.STEM_COUNT] = entries[0][formats.STEM_COUNT]
				record[formats.SPECIES] = entries[0][formats.SPECIES]

				record[formats.DW_PLANT] = entries[1][formats.DW_PLANT]

				if record[formats.DW_PLANT] > entries[1][formats.FW_PLANT] or \
					record[formats.DW_PLANT] == None:
					dropped += 1
					continue

				final_data.append(record)

		if dropped > 0:
			print "WARNING! Dropped %d records, because dry_weight > fresh_weight!" % dropped

		return final_data

	def get_JKI_data(self):
		location = Location(Location.JKI15, -1000)

		#bypass the normal result obtaining method as we need the species info as well
		reader = PhenoReaderMSCAN(location)
		records = reader.parse_result(reader._query, include_species = True) 
		
		#select 2nd year records for JKI 15 
		records = self.select_records(datetime(2014, 10, 1), datetime(2015, 05, 01), records)
		return records





	### PROCESS BSBEC DATA ###

	def get_BSBEC_data(self, locations):
		bsbec_data = self.read_BSBEC_data(locations)
		bsbec_data = self.add_leaf_area(bsbec_data)
		bsbec_data = self.format_BSBEC(bsbec_data)
		return(bsbec_data)


	def format_BSBEC(self, bsbec_data):
		"""Grab the stem_count, canopy_height and dry_matter measurements from the BSBEC data"""
		#remove the leaf measurements
		entries = [x for x in bsbec_data if x['pseudo_rep_no'] in [None, 0]]
		
		#calculate the dry matter yield
		for entry in entries:
			if entry[formats.FW_PLANT] != None and entry[formats.FW_SUB] != None and\
				entry[formats.DW_SUB] != None:
				fw = entry[formats.FW_PLANT]
				fw_sub = entry[formats.FW_SUB]
				dw_sub = entry[formats.DW_SUB]
				entry[formats.DW_PLANT] = fw * (dw_sub / fw_sub)


		#this extracts the stem_count, canopy_height and dry_weight measurements 
		#and puts them into a corresponding list in the plot_measurements dictionary
		plot_measurements = {formats.CANOPY_HEIGHT: [], 
							formats.STEM_COUNT: [], 
							formats.LEAF_AREA: [], 
							formats.DW_PLANT: []}

		for entry in entries:
			for key in plot_measurements.keys():
				if key in entry.keys() and entry[key] != None:
					#make sure the measurements here are plot level measurements, 
					#i.e. they all end on 0
					if key == formats.LEAF_AREA:
						#leaf area measurements get assigned to plot level here
						uid = entry[formats.UID] - entry[formats.UID]%10
					else:
						uid = entry[formats.UID]

					plot_measurements[key].append({formats.UID: uid,
											formats.DATE: entry[formats.DATE],
											key: entry[key]})


		#take the average of the plot level measurements
		final_data = dict()
		for key in plot_measurements.keys():
			final_data[key] = self.get_average(plot_measurements[key], key)

		return final_data


	def read_BSBEC_data(self, locations):
		pheno_data = self.bind_data(locations, PhenoReaderCSV)
		yield_data = self.bind_data(locations, YieldReaderCSV)
		return pheno_data + yield_data



	### INJECT RECORDS AND INTERPOLATE ####
	def inject_records(self, original_data):
		"""
		Inserts 0 reading on FLL day and copies last pheno measurement to the last harvest day for
		the PHENO_VARS (stem_count and canopy_height)."""
		data = copy.deepcopy(original_data)

		for key in self.PHENO_VARS:
			uids = uniq_uids(data[key])

			for uid in uids:
				plot_data = [x for x in data[key] if x[formats.UID] == uid]
				years = set([x[formats.DATE].year for x in plot_data])
	
				#for each season 	
				for year in years:
					location = next(x for x in self._locations if x.get_year() == year)
					fll_read = FLLReader(location)
					season_start = fll_read.get_plot_fll(uid/10)

					#the latest possible date for a harvest of this season is actually the
					#following year - Jan, Feb final harvest
					season_max = datetime(year + 1, 3, 1) 
					
					harvest_data = [x for x in data[formats.DW_PLANT] if 
													x[formats.DATE] >= season_start and
													x[formats.DATE] <= season_max and 
													str(x[formats.UID])[:2] == str(uid)[:2]]
					#actual season end is the last harvest for that plot
					season_end = max([x[formats.DATE] for x in harvest_data]) #TODO hack
					#season_end = datetime(year, 9, 6)

					#now insert a 0 in the begining of the season
					data[key].append({formats.UID: uid,
										formats.DATE: season_start,
										key: 0.0})
	
					#finally copy the last measurement across to the end of the season
					#so that we have no problems when exterpolating 
					last_measurement_date = max([x[formats.DATE] for x in plot_data if 
																	x[formats.DATE] < season_end])
					last_measurement = next(x for x in plot_data if 
														x[formats.DATE] == last_measurement_date)

					data[key].append({formats.UID: uid,
										formats.DATE: season_end,
										key: last_measurement[key]})

		return data


	def interpolate(self, data):
		interpolated_data = []
		years = [x.get_year() for x in self._locations]
		uids = uniq_uids(data[formats.DW_PLANT])

		for year in years:
			#rough guidelines of where one growth year starts and ends
			start_date = datetime(year, 3, 2)
			end_date = datetime(year + 1, 3, 1)
			for uid in uids:
				plot_data = dict()
				splines = dict()
				for key in self.PHENO_VARS:
					plot_data[key] = [x for x in data[key] if x[formats.UID] == uid and 
																x[formats.DATE] >= start_date and 
																x[formats.DATE] <= end_date]

					plot_data[key].sort(key = lambda x: x[formats.DATE])
					
					#build interpolation function
					x_var = [time.mktime(x[formats.DATE].timetuple()) for x in plot_data[key]]
					y_var = [x[key] for x in plot_data[key]]
					splines[key] = interp1d(x_var, y_var)

				season_start = min([x[formats.DATE] for x in plot_data[plot_data.keys()[0]]])
				season_end = max([x[formats.DATE] for x in plot_data[plot_data.keys()[0]]]) 
				
				#interpolate between the actual season start and end dates
				current_date = season_start
				while current_date <= season_end:
					entry = {formats.UID: uid,
							formats.DATE: current_date}

					for key in self.PHENO_VARS:
						entry[key] = splines[key](time.mktime(current_date.timetuple()))

					interpolated_data.append(entry)
					current_date += timedelta(1)

		return interpolated_data
			

	### MISC FUNCTIONS ###


	def select_records(self, min_date, max_date, records):
		return [x for x in records if x[formats.DATE] >= min_date and x[formats.DATE] <= max_date]

	def bind_data(self, locations, reader_class):
		final_data = []
		for location in locations:
			final_data += reader_class(location).get_records()
		
		return final_data

	def get_average(self, entries, key):
		"""Obtains average values for <key> for every EuroPheno.stock_id + pheno_date combination"""
		uids = set([x[formats.UID] for x in entries])
		dates = set([x[formats.DATE] for x in entries])
	
		final_data = []
		for uid in uids:
			for date in dates:
				new_entry = {formats.UID: uid, formats.DATE: date}
				match = [x for x in entries if x[formats.UID] == uid and \
												x[formats.DATE] == date]
				if len(match) == 0:
					continue 

				total = 0

				for entry in match:
					total += entry[key]

				total = total / len(match)
				new_entry[key] = total
				final_data.append(new_entry)

		return final_data

	def attach_geno_data(self, data):
		for entry in data:
			uid = entry[formats.UID]
			entry['genotype'] = assign_geno_bsbec(uid/10)
			entry['species'] = get_species(uid/10)
		
		return data

	def add_leaf_area(self, data):
		#add dummy leaf_area variable everywhere
		for entry in data:
			entry[formats.LEAF_AREA] = None
		
		uids = set(x[formats.UID] for x in data if x[formats.UID] % 10 > 0)
		for uid in uids:
			plant_data = [x for x in data if x[formats.UID] == uid]
			dates = sorted(set(x[formats.DATE] for x in plant_data))

			prev_leaves = dict()
			stem_id = 1
			prev_date = None

			for date in dates:
				#detect year change
				if prev_date and date.year > prev_date.year:
					prev_leaves = dict()
					stem_id = 1
				
				prev_date = date

				day_data = [x for x in plant_data if x[formats.DATE] == date]
				try:
					head_entry = next(x for x in day_data if x[formats.PSEUDO_REP] == 0)
				except:
					continue

				leaf_entries = [x for x in day_data if x[formats.PSEUDO_REP] > 0]
				leaf_entries.sort(key = lambda x: x[formats.PSEUDO_REP])

				#detect stem change
				if head_entry[formats.STEM_ID] != stem_id:
					stem_id = head_entry[formats.STEM_ID]
					prev_leaves = dict()

				area = 0

				#add up all the previous leaves that have not been updated
				updated = [x[formats.PSEUDO_REP] for x in leaf_entries]
				for key in prev_leaves.keys():
					if not key in updated:
						area += prev_leaves[key][formats.LEAF_WIDTH] * \
								prev_leaves[key][formats.LEAF_LENGTH]


				for leaf in leaf_entries:
					#update the record
					prev_leaves[leaf[formats.PSEUDO_REP]] = leaf

					area += leaf[formats.LEAF_WIDTH] * leaf[formats.LEAF_LENGTH]
						

				head_entry[formats.LEAF_AREA] = area

		return data

#	#Temporary method that should be removed once you have a better way of outputting 
#	#validation data TODO
#	def output_validation_data(self, location):
#		#THIS METHOD IS A COMPLETE HACK TODO
#		data = self.read_BSBEC_data([location])
#		data = self.add_leaf_area(data)
#		data = self.format_BSBEC(data)
#		
#		#fix uids
#		for key in data.keys():
#			for entry in data[key]:
#				entry[formats.UID] = entry[formats.UID] - entry[formats.UID]%10
#
#		self._locations = [location]
#		data_interp = self.interpolate(data)
#		data_interp = self.attach_geno_data(data_interp)
#		yield_data = self.attach_geno_data(data[formats.DW_PLANT])
#
#		CSVFileWriter("/home/eey9/Scratch/ann/validation.csv", data_interp)
#		CSVFileWriter("/home/eey9/Scratch/ann/validation_yield.csv", yield_data)
