# The frontend of Chris Davey's model
from framework.data.phenotype.pheno_reader.pheno_reader_csv import PhenoReaderCSV
from framework.data.phenotype.sunscan_reader.transmission_reader import TransmissionReader
from framework.data.phenotype.pheno_reader.fll_reader import FLLReader
from framework.data.met_data.metdata_reader_csv import MetDataReaderCSV

from framework.util.csv_io.csv_filewriter import CSVFileWriter
from framework.util.csv_io.csv_filereader import CSVFileReader

from framework.models.ler_coefficient import LERCoefficient
from framework.models.degree_days import degree_days
from framework.models.lm import LM
from framework.models.LAI_plateau_detector import LAIPlateauDetector

import numpy
from scipy.interpolate import interp1d
import tempfile
import math
from datetime import datetime
from datetime import timedelta
import time
import copy 

from rpy2 import robjects

class ChrisDaveyModel:
	def __init__(self, locations):
		self._locations = locations
	
		#stage one - parameterise LER and get LAI data for all years
		plots = []
		transmission = []
		for location in self._locations:
			#parameterise LER on the first year of data
			calc_LER = not self._locations.index(location)
			plots += self.prep_plots(location, calc_LER)

			#grab the transmission data - useful for k parameterisation
			transmission += TransmissionReader(location).get_data()


		#stage two - parameterise k
		interp_transmissions = self.interpolate_transmissions(transmission, plots)
		LAI_transmiss = self.mean_transmission_LAI(interp_transmissions)
		self._k = self.calc_k(LAI_transmiss)		

		#run the simulation in order to get RUE
		LAI_transmiss = [x for x in LAI_transmiss if int(x['Date'].strftime("%Y")) == 
														locations[0].get_year()]
		self._RUE = self.calc_RUE(self._LER, self._k, 
								self._locations[0], LAI_transmiss)


	def simulate(   self, location, genotypes, DM_repartition = False, 
					harvest_date = None, end_date = None):

		met_data = MetDataReaderCSV(location).get_met_data()
		result = []
		year = location.get_year()
		fll_reader = FLLReader(self._locations[0])
		plants = PhenoReaderCSV(location, "eey9").get_plants()

		for genotype in genotypes:
			geno_simulation = [] 

			geno_plants = [x for x in plants if x.get_genotype() == genotype]
			if not end_date:
				end_date = LAIPlateauDetector(geno_plants, location, genotype).get_plateau()

			#set end date as the LAI plateau point
			geno_plants = [x for x in plants if x.get_genotype() == genotype]
			
			#get genotype specific coefficients
			LER = [x for x in self._LER if x['Genotype'] == genotype]
			k = [x for x in self._k if x['Genotype'] == genotype]
			#take the second year k value
			if len(k) > 1:
				k = k[1]['k']
			else:
				k = k[0]['k']

			RUE = next(x for x in self._RUE if x['Genotype'] == genotype)['RUE']

			#change the fll year to the current year
			current_date = fll_reader.get_genotype_fll(genotype).replace(year = year)
			current_date += timedelta(days = 1)
			LAI = 0
			dd = 0
			PAR = 0
			DMY = 0
			while current_date <= end_date:
				geno_simulation.append({'Date': current_date,
								'Genotype': genotype,
								'dd': dd,
								'LAI': LAI,
								'PAR': PAR,
								'Yield': DMY})
	
				#delta dd
				met_day = next(x for x in met_data if x['Date'] ==
														current_date)

				delta_dd = degree_days(met_day['t_max'], met_day['t_min'], 0.0)

				#delta LAI
				if len(LER) == 1:
					current_LER = LER[0]['LER']
				else:
					if dd < LER[0]['psi']:
						current_LER = LER[0]['LER']
					else:
						current_LER = LER[1]['LER']
	
				delta_LAI = delta_dd * current_LER
				
				#delta PAR
				pl = 1 - math.exp(-k * LAI)
				delta_PAR = met_day['PAR'] * pl
		
				#delta DMY
				delta_DMY = delta_PAR * RUE

				dd += delta_dd
				LAI += delta_LAI
				PAR += delta_PAR
				DMY += delta_DMY
				current_date += timedelta(days = 1)
				

			if DM_repartition:
				final_yield = self.repartition_DM(geno_simulation, harvest_date)
				geno_simulation.append({'Date': harvest_date, 
									'Genotype': genotype,
										'dd': -999,
										'LAI': -999,
										'PAR': -999,
										'Yield': final_yield})

			result.append(geno_simulation)

		return result


	def prep_plots(self, location, calc_LER):
		"""
		Calculates the LAI values per plant. Organises the data in plots.
		If calc_LER == True the LER coefficient will be parameterised on 
		this data in which case it will overwrite the self._LER dictionary!
		"""
		reader = PhenoReaderCSV(location, "eey9")
		plants = reader.get_plants()

		met_reader = MetDataReaderCSV(location)
		met_data = met_reader.get_met_data()

		coeff = LERCoefficient(plants, met_data, location, calc_LER)

		if calc_LER:
			self._LER = coeff.get_average_LER()
		
		return coeff._plots


	def interpolate_transmissions(self, transmission, plots):
		"""
		Interpolate transmissions values to match the measurement dates
		"""
		result = []
		for plot in plots:
			measurements = plot.get_mean_measurements()
			year = measurements[0]['Date'].strftime("%Y")
			plot_transmission = [x for x in transmission
										if x['Plot'] == plot.get_UID() and
											x['Year'] == year]

			plot_transmission.sort(key = lambda x: x['Date'])

			#assume a last measurement so that we can interpolate until the end
			plot_transmission.append({'Date': measurements[-1]['Date'],
									'Day': int(measurements[-1]['Date'].strftime("%j")),
									'Plot': plot.get_UID(),
									'Transmission': plot_transmission[-1]['Transmission'],
									'Year': year})

			#add first 0 0 measurement on FLL day
			plot_transmission.append({'Date': measurements[0]['Date'],
									'Day': int(measurements[0]['Date'].strftime("%j")),
									'Plot': plot.get_UID(),
									'Transmission': 1.0,
									'Year': year})
			
			plot_transmission.sort(key = lambda x: x['Date'])

			days = numpy.array([x['Day'] for x in plot_transmission])
			transmission_vals = numpy.array([x['Transmission'] for x in plot_transmission])
			interp_transmiss = interp1d(days, transmission_vals)
			for measurement in measurements:
				day = int(measurement['Date'].strftime("%j"))
				measurement['Transmission'] = float(interp_transmiss(day))
				measurement['Genotype'] = plot.get_genotype()

			result += measurements

		return result

	def mean_transmission_LAI(self, measurements):
		"""
		Works with measurements containing interpolated transmission values,
		see interpolate_transmissions. Calculates the mean LAI and transmission
		values per genotype.
		"""
		dates = set([x['Date'] for x in measurements])
		genotypes = set([x['Genotype'] for x in measurements])
		result = []
		for date in dates:
			for genotype in genotypes:
				subset = [x for x in measurements if x['Date'] == date and
													x['Genotype'] == genotype]
				if subset:
					LAI_values = [x['LAI'] for x in subset]
					LAI_avg = sum(LAI_values)/len(LAI_values)
				
					transmiss_values = [x['Transmission'] for x in subset]
					transmiss_avg = sum(transmiss_values)/len(transmiss_values)

					result.append({'Genotype': genotype,
									'Date': date,
									'Year': date.strftime("%Y"),
									'LAI': LAI_avg,
									'Transmission': transmiss_avg})

		result.sort(key = lambda x: (x['Genotype'], x['Date']))
		return result

	def calc_k(self, LAI_transmiss):
		temp_file = tempfile.mkstemp()[1]
		CSVFileWriter(temp_file, LAI_transmiss)

		robjects.r('''
			calc_k_sep_r <- function(fname, genotype){
				data <- read.csv(fname)
				data$Date <- as.Date(data$Date, "%Y-%m-%d %H:%M:%S")

				geno_sub <- subset(data, Genotype == genotype)
				years <- levels(factor(geno_sub$Year))
				year_subs <- lapply(years, function(year) subset(geno_sub, Year == year))
				k <- sapply(year_subs, function(year_sub)
								summary(nls(Transmission ~ exp(-k * LAI), 
										data = year_sub, 
										start = list(k = 1)))$parameters[1])

				return(data.frame(as.integer(years), k))
			}
			''')

		robjects.r('''
			calc_k_r <- function(fname, genotype){
				data <- read.csv(fname)
				data$Date <- as.Date(data$Date, "%Y-%m-%d %H:%M:%S")

				geno_sub <- subset(data, Genotype == genotype)
				k <- summary(nls(Transmission ~ exp(-k * LAI), 
										data = geno_sub, 
										start = list(k = 1)))$parameters[1]

				return(k)
			}
			''')

		calc_k_sep_r = robjects.r("calc_k_sep_r")
		calc_k_r = robjects.r("calc_k_r")

		genotypes = set([x['Genotype'] for x in LAI_transmiss])
		result = []
		for genotype in genotypes:
			df = calc_k_sep_r(temp_file, genotype)
			diff = math.fabs(df[1][0] - df[1][1])
			if diff > 0.3:
				for year, k in zip(df[0], df[1]):
					result.append({'Genotype': genotype, 'Year': year, 'k': k})

			else:
				#bulk the two years together
				k = calc_k_r(temp_file, genotype)[0]
				result.append({'Genotype': genotype, 'Year': '', 'k': k})

		return result

	def calc_RUE(self, LER_dict, k_dict, location, LAI):
		met_data = MetDataReaderCSV(location).get_met_data()
		fll_reader = FLLReader(location)
		genotypes = set([x['Genotype'] for x in LER_dict])
		
		destructive_phenos = CSVFileReader(location.get_destr_phenos()).get_content()
		for entry in destructive_phenos:
			entry['Date'] = datetime.strptime(entry['Date'], "%Y-%m-%d %H:%M:%S UTC")

			try:
				entry['fresh'] = float(entry['Fresh weight above ground material(g)'])
				entry['fresh_sub'] = float(entry['Fresh weight above ground  sub-sample(g)'])
				entry['dry_sub'] = float(entry['dry weight above ground sub-sample(g)'])
			except ValueError:
				try:
					entry['dry_weight'] = float(entry['dry weight above ground sub-sample(g)'])
				except ValueError:
					pass
				continue

			if entry['fresh_sub'] == 0.0:
				entry['dry_weight'] = entry['dry_sub']
				continue
			entry['dry_weight'] = entry['fresh'] * (entry['dry_sub']/entry['fresh_sub'])
		
		destructive_phenos = [x for x in destructive_phenos if 'dry_weight' in x]

		#run the simulation per genotype
		RUE = []
		for genotype in genotypes:
			geno_sub = [x for x in destructive_phenos if x['Genotype'] == genotype]
			dates = list(set([x['Date'] for x in geno_sub]))
			dates.sort()

			#create data point groups by dates that are close 
			#to each other or the same
			groups = []
			group_id = 0
			for date in dates:
				for group in groups:
					delta = group['Dates'][0] - date
					days = math.fabs(delta.days)
					if days and days < 20:
						group['Dates'].append(date)
						break
				else:
					#create new group
					group = {'id': group_id,
							'Dates': [date]}
					groups.append(group)
					group_id += 1

			#get the mean dry weight per group
			mean_DW = []
			#add entry for fll day
			fll_date = fll_reader.get_genotype_fll(genotype)
			mean_DW.append({'Date': fll_date, 'Yield': 0.0})

			for group in groups:
				group_phenos = [x for x in geno_sub if x['Date'] in
													group['Dates']]
				total_dw = 0.0
				for entry in group_phenos:
					total_dw += entry['dry_weight']

				total_dw /= float(len(group_phenos))

				#correct the group date to the first one in the group
				mean_DW.append({'Date': sorted(group['Dates'])[0],
								'Yield': total_dw}) 
			
			#obtain genotype specific coefficients
			LER = [x for x in LER_dict if x['Genotype'] == genotype]
			LER.sort(key = lambda x: x['stage'])
			k = [x for x in k_dict if x['Genotype'] == genotype]
			if len(k) > 1:
				k = next(x['k'] for x in k if x['Year'] == location.get_year())
			else:
				k = sorted(k, key = lambda x: x['Year'])[0]['k']
			
			#simulate PAR and record values for days of destructive harvests
			real_LAI = [x for x in LAI if x['Genotype'] == genotype]
			mean_DW = self.simulate_PAR(k, LER, met_data, fll_date, mean_DW, real_LAI)
	
			#finally work out what the RUE is from 
			#the real DMY and simulated PAR values
			temp_file = tempfile.mkstemp()[1]+genotype.split("-")[0]
			CSVFileWriter(temp_file, mean_DW)
			robjects.r('''
				calc_RUE_r <- function(fname){
					data <- read.csv(fname)
					data$Yield <- data$Yield * 2
					fit <- lm(Yield ~ PAR + 0, data = data)
					return(summary(fit)$coefficients[1])
				}
				''')
			calc_RUE_r = robjects.r("calc_RUE_r")
			RUE_val = calc_RUE_r(temp_file)[0]
			RUE.append({'Genotype': genotype, 'RUE': RUE_val})

		return RUE

	def simulate_PAR(self, k, LER, met_data, fll_date, mean_DW, real_LAI):
		mean_DW = copy.deepcopy(mean_DW)
		current_date = fll_date
		LAI = 0
		dd = 0
		PAR = 0
		destr_dates = set([x['Date'] for x in mean_DW])
		
		sim_record = [] #a recording of the simulation values
		while current_date <= max(destr_dates):
			sim_record.append({'Date': current_date, 'dd': dd, 
								'LAI': LAI, 'PAR': PAR})

			#delta dd
			met_day = next(x for x in met_data if x['Date'] == current_date)
			delta_dd = degree_days(met_day['t_max'], met_day['t_min'], 0.0)
			
			#delta LAI
			if len(LER) == 1:
				current_LER = LER[0]['LER']
			else:
				if dd < LER[0]['psi']:
					current_LER = LER[0]['LER']
				else:
					current_LER = LER[1]['LER']

			delta_LAI = delta_dd * current_LER
			
			#delta PAR
			pl = 1 - math.exp(-k * LAI)
			delta_PAR = met_day['PAR'] * pl
	
			dd += delta_dd
			LAI += delta_LAI
			PAR += delta_PAR
			
			if current_date in destr_dates:
				dw_entry = next(x for x in mean_DW if x['Date'] == current_date)
				dw_entry['PAR'] = PAR

			current_date += timedelta(days = 1)

		#interpolate the real LAI values
		real_LAI = copy.deepcopy(real_LAI)
		for entry in real_LAI:
			entry['Day'] = int(entry['Date'].strftime("%j"))

		days = numpy.array([x['Day'] for x in real_LAI])
		lai = numpy.array([x['LAI'] for x in real_LAI])
		interp_LAI = interp1d(days, lai)
		last_LAI = real_LAI[-1]

		breakpoint = self.detect_LAI_breakpoint(real_LAI, sim_record)

		#correct the simulation values for LAI with the real ones after the breakpoint
		#and recalculate the PAR
		next_PAR = None
		for entry in sim_record:
			if breakpoint and entry['Date'] >= breakpoint['Date']:
				if entry['Date'] <= last_LAI['Date']:
					if next_PAR:
						entry['PAR'] = next_PAR

					met_day = next(x for x in met_data if x['Date'] == entry['Date'])
					LAI = interp_LAI(int(entry['Date'].strftime("%j")))
					next_PAR = met_day['PAR'] * (1 - math.exp(-k * LAI)) + entry['PAR']

				else:
					break

			#also add the PAR if the date is in destructive measurements
			destr_measurement = [x for x in mean_DW if x['Date'] == entry['Date']]
			if destr_measurement:
				destr_measurement[0]['PAR'] = entry['PAR']

		return mean_DW


	def detect_LAI_breakpoint(self, real_LAI, sim_record):
		breakpoint = None
		while not breakpoint and real_LAI:
			#check the slope of the real LAI
			x_data = [x['Day'] for x in real_LAI]
			y_data = [y['LAI'] for y in real_LAI]
			beta = LM(x_data, y_data)._beta
			entry = real_LAI.pop(0)

			if beta < 0.025:
				sim_entry = next(x for x in sim_record if x['Date'] == entry['Date'])
				diff = sim_entry['LAI'] - entry['LAI']
				if diff > 0.2:
					breakpoint = entry
					break

		return breakpoint


	def repartition_DM(self, simulation, harvest_date):
		#use linear interpolation to simulate value at harvest
		#1 March is 0.66 of the peak yield in MISCANFOR
		start_date = simulation[-1]['Date']
		end_date = start_date.replace(year = start_date.year + 1)
		end_date = end_date.replace(day = 1, month = 3)
	
		#now convert into unix timestamp so that we can use them in interp1d
		start_date = time.mktime(start_date.timetuple())
		end_date = time.mktime(end_date.timetuple())

		dates = numpy.array([start_date, end_date])
		yields = numpy.array([simulation[-1]['Yield'], 0.66 * simulation[-1]['Yield']])
		interp_yield = interp1d(dates, yields)
		return float(interp_yield(time.mktime(harvest_date.timetuple())))
