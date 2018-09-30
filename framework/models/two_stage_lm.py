from framework.util.csv_io.csv_filewriter import CSVFileWriter
from framework.util.misc import uniq_uids
import framework.data.formats as formats

import time
import datetime
import os, tempfile
import copy
from collections import defaultdict
import rpy2.robjects as robjects

class TwoStageLM:
	"""
	TwoStageLM is a yield estimation model that can be used to fill the yield data gaps for the 
	2011 and 2012 BSBEC data. 
	train_data - data from another field trial, e.g. JKI 15, which is used to build generic models.
	train_data is a list of dicts, each dict contains the following keys:
	'EuroPheno.stock_id', 'species', 'canopy_height', 'stem_count', 'dry_weight'a
	test_data - data from BSBEC, the generic model will predict yield on it and the specific model
	will be trained on it
	yield_data - real yield data from BSBEC"""

	def __init__(self, train_data, test_data, yield_data):
		self.define_r_functions()
		self._train_data = train_data
		self._test_data = test_data
		self._yield_data = yield_data

		self._final_data = []
	
	def gap_fill_data(self, genotype, species):
		self._final_data += self.gap_fill(self._train_data, self._test_data, self._yield_data, 
											genotype, species)

	### THE ESSENTIAL FUNCTION 
	def gap_fill(self, train_data, data, yield_data, genotype, species):
		train_data = copy.deepcopy(train_data)
		data = copy.deepcopy(data)
		yield_data = copy.deepcopy(yield_data)

		#subset the training_data
		if type(species) == str:
			train_data = [x for x in train_data if x[formats.SPECIES] == species]
		else: 
			train_data = [x for x in train_data if x[formats.SPECIES] in species]

		data = [x for x in data if x[formats.GENOTYPE] == genotype]
		yield_data = [x for x in yield_data if x[formats.GENOTYPE] == genotype]


		#convert all to data frames
		train_data = self.to_dataframe(train_data)
		data = self.to_dataframe(data)
		yield_data = self.to_dataframe(yield_data)

		
		#insert the data frames into the environment
		robjects.globalenv['train_data'] = train_data
		robjects.globalenv['data'] = data
		robjects.globalenv['yield_data'] = yield_data


		#convert data to the right formats in R 
		robjects.r("""
			data <- format_data(data)
			yield_data <- format_data(yield_data)
		""")

	
		#scale training data
		for pheno in [formats.STEM_COUNT, formats.CANOPY_HEIGHT, formats.DW_PLANT]:
			robjects.r("train_data$%s <- scale(train_data$%s)" % (pheno, pheno))

		
		#train the generic model
		robjects.r("generic_model <- lm(%s ~ %s + %s, data = train_data)" % (formats.DW_PLANT,
																			formats.STEM_COUNT,
																			formats.CANOPY_HEIGHT))


		#Scale test data and make generic predictions of yield
		for pheno in [formats.STEM_COUNT, formats.CANOPY_HEIGHT]:
			robjects.r("data[['%s']] <- scale_by(data, train_data, '%s')" % (pheno, pheno))
		
		#scale leaf area on its own - will be used when training the specific model
		robjects.r("data$%s <- scale(data$%s)" % (formats.LEAF_AREA, formats.LEAF_AREA))

		#make a generic prediction
		robjects.r("data$generic_yield <- predict(generic_model, data)")
		

		#scale the real yield and train the specific model
		robjects.r("yield_data$%s <- scale_by(yield_data, train_data, '%s')" % (formats.DW_PLANT,
																			formats.DW_PLANT))
	
		robjects.r("specm_tdata <- merge(data, yield_data, by = c('%s', '%s', '%s', '%s'))" % 
									(formats.UID, formats.DATE, formats.GENOTYPE, formats.SPECIES))

		robjects.r("specm_tdata$doy <- as.numeric(strftime(specm_tdata$%s, '%%j'))" % formats.DATE)
		robjects.r("specm_tdata$doy <- scale(specm_tdata$doy)")

		robjects.r("specific_model <- lm(%s ~ generic_yield + %s + %s + %s + %s,"\
															" data = specm_tdata)" % 
											(formats.DW_PLANT, formats.STEM_COUNT, 
											formats.CANOPY_HEIGHT, formats.LEAF_AREA, 'doy'))

		
		
		#make predictions using the specific model
		robjects.r("data$doy <- scale(as.numeric(strftime(data$%s, '%%j')))" % formats.DATE)
		robjects.r("data$%s <- predict(specific_model, data)" % formats.DW_PLANT)
		robjects.r("data$doy <- NULL")

		#unscale stuff
		for pheno in [formats.STEM_COUNT, formats.CANOPY_HEIGHT, formats.DW_PLANT]:
			robjects.r("data$%s <- unscale_by(data, train_data, '%s')" % (pheno, pheno))

		#leaf area is unscaled by itself as we do not have train_data$leaf_area column
		robjects.r("data$%s <- unscale_by(data, data, '%s')" % (formats.LEAF_AREA, 
																formats.LEAF_AREA))
		
		#get rid of useless data
		robjects.r("data$generic_yield <- NULL")
		
		#finally convert back to a normal and nice list of dicts 
		data = self.to_listdict("data")

		return data


	def to_dataframe(self, data):
		#transpose the list of dicts into a dict of lists
		transposed = defaultdict(list)
		for entry in data:
			for key in entry:
				transposed[key].append(entry[key])

		
		for key in transposed:
			#get the appropriate conversion function
			if key in formats._INTS:
				conv_f = robjects.IntVector
			elif key in formats._FLOATS:
				conv_f = robjects.FloatVector
			elif key == formats.DATE:
				transposed[key] = formats.strfdate_standard_list(transposed[key])
				conv_f = robjects.StrVector
			elif key in formats._NO_CONVERSION:
				conv_f = robjects.StrVector
			else:
				raise Exception("Conversion function not found for key %s" % key)

			transposed[key] = conv_f(transposed[key])
		
		data_frame = robjects.DataFrame(transposed)
		return data_frame

	def to_listdict(self, data_str):
		#convert to python format
		phenos = [formats.UID, formats.DATE, formats.GENOTYPE, formats.SPECIES]
		for pheno in phenos:
			if pheno == formats.DATE:
				robjects.r("%s$%s <- format(%s$%s, '%%d/%%m/%%Y')" % (data_str, pheno, 
																		data_str, pheno) )
			else:
				robjects.r("%s$%s <- as.character(%s$%s)" % (data_str, pheno, data_str, pheno))

		data = robjects.r("%s" % data_str)

		keys = data.names
		output = []

		for i in range(data.nrow):
			entry = dict()
			for key in keys:
				entry[key] = data[data.names.index(key)][i]
		
			entry = formats.on_read(entry, just_convert = True)
			output.append(entry)
		

		return output

	def define_r_functions(self):
		robjects.r("""
				format_data <- function(data){
					data$%s <- as.Date(data$%s, "%%d/%%m/%%Y")
					data$%s <- factor(data$%s)
					return(data)
				}
				""" % (formats.DATE, formats.DATE, formats.UID, formats.UID))
		
		robjects.r("""
				scale_by <- function(data, example, colname){
					center = attr(example[[colname]], "scaled:center")
					scale = attr(example[[colname]], "scaled:scale")
					result <- scale(data[[colname]], center, scale)
					return(result)
				}
				""")


		robjects.r("""
				unscale_by <- function(data, example, colname){
					center = attr(example[[colname]], "scaled:center")
					scale = attr(example[[colname]], "scaled:scale")
					return(data[[colname]] * scale + center)
				}
				""")
