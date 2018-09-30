import copy
from datetime import timedelta
from framework.models.plot import Plot
from rpy2 import robjects
from rpy2.rinterface import RRuntimeError

from scipy import stats

from framework.data.phenotype.pheno_reader.fll_reader import FLLReader
from framework.models.plateau_detector import PlateauDetector
from framework.models.degree_days import degree_days

class LERCoefficient:
	
	def __init__(self, plants, met_data, location, calculate_LER = True, threshold = 0.00075):
		"""
		plants - needs to be a list of plants of the same genotype
		"""
		self._threshold = threshold
		self._plants = copy.deepcopy(plants)
		self._genotype = plants[0].get_genotype()
		self._met_data = met_data
		self._location = location

		#LER estimation phase
		self._plots = self.create_plots()
		self._noplateau_plots = self.remove_plateau(self._plots)

		if calculate_LER:
			self._LER = self.calc_LER(self._noplateau_plots)
			self._avg_LER = self.calc_average_LER(self._LER)

	def create_plots(self):
		plot_names = set([x.get_plot() for x in self._plants])
		plots = []
		for plot_name in plot_names:
			plots.append(self.create_plot(plot_name))

		return plots


	def create_plot(self, plot_n):
		"""
		Create plot object and populate it with the LAI and dd measurements
		"""
		plants = [plant for plant in self._plants if plant.get_plot() == plot_n]
		genotype = plants[0].get_genotype()
		leaf_conversion = self._location.get_leaf_conversion(genotype)

		fll_date = FLLReader(self._location).get_plot_fll(plot_n)
		pheno_dates = [measurement.get_date() for measurement in
						plants[0].get_pheno_measurements()]
		weekly_dd = self.get_weekly_dd(pheno_dates, self._met_data, fll_date)

		plot = Plot(plot_n, genotype)
		#first liguled leaf stage - LAI and dd are 0
		for plant in plants:
			plot.add_measurement(plant.get_uid(), 0.0, 0.0, fll_date)
			for measurement in plant.get_pheno_measurements():
				LAI = self.calc_LAI(measurement, leaf_conversion)
				dd = next(x['dd'] for x in weekly_dd if x['Date'] == measurement.get_date())
				plot.add_measurement(plant.get_uid(), LAI, dd, measurement.get_date())

		return plot


	def remove_plateau(self, plots):
		editted_plots = copy.deepcopy(plots)
		for plot in editted_plots:
			#get corresponding plot
			measurements = plot.get_mean_measurements()

			#sort measurements by date
			measurements.sort(key = lambda x: x['Date'])

			plateau_input = []
			for measurement in measurements:
				entry = dict()
				entry['x'] = measurement['dd']
				entry['y'] = measurement['LAI']
				plateau_input.append(entry)

			plateau = PlateauDetector(plateau_input, self._threshold)
			if plateau.get_plateau() != None:
				plot.set_measurements([x for x in plot.get_measurements() 
											if x['dd'] <= plateau.get_plateau()])
	
		return editted_plots
		
		
	def get_LER(self):
		return self._LER_vals

	def calc_LER(self, plots):
		LER_vals = []
		for plot in plots:
			measurements = plot.get_measurements()
	
			#run LM LAI ~ dd, LER is the slope
			x = [a['dd'] for a in measurements]
			y = [b['LAI'] for b in measurements]
			
			x_r = robjects.FloatVector(x)
			y_r = robjects.FloatVector(y)

			#define the two types of functions (segmented and lm)
			robjects.r('''
				get_LER_segmented <- function(x,y){
					require("segmented")
					m <- mean(x)
					segment <- segmented(lm(y ~ x + 0), seg.Z=~x, psi=m)
					return(c(slope(segment)$x[1], slope(segment)$x[2], segment$psi[2], 
								summary(segment)$adj.r.squared))
				}
				''')

			robjects.r('''
				get_LER_lm <- function(x,y){
					fit <- lm(y ~ x + 0)
					result <- summary(fit)
					return(c(result$coefficients[1], result$adj.r.squared))
				}
				''')
					

			get_LER_lm = robjects.r('get_LER_lm')
			get_LER_segmented = robjects.r('get_LER_segmented')

			while True:
				try:
					if plot.get_genotype() in ['EMI-11', 'Giganteus']:
						LER = get_LER_lm(x_r, y_r)
					else:
						LER = get_LER_segmented(x_r, y_r)
					
					break;
				except RRuntimeError:
					pass
			
			LER_vals += self.build_LER_dicts(LER, plot)

		return LER_vals
	
	def build_LER_dicts(self, LER, plot):
		"""
		Organises the output from LER R function into list of dicts.
		"""
		dicts = []
		if len(LER) == 4:
			dicts.append({'LER': LER[0],
						'psi': LER[2],
						'Plot': plot.get_UID(),
						'Genotype': plot.get_genotype(),
						'stage': 1})

			dicts.append({'LER': LER[1],
						'psi': False,
						'Plot': plot.get_UID(),
						'Genotype': plot.get_genotype(),
						'stage': 2})
		elif len(LER) == 2:
			dicts.append({'LER': LER[0],
						'psi': False,
						'Plot': plot.get_UID(),
						'Genotype': plot.get_genotype(),
						'stage': 1})

		else:
			raise Exception("Cannot handle output from LER function!")
	
		return dicts

	def calc_LAI(self, measurement, leaf_conversion):
		leaves = measurement.get_leaves()
		stem_leaf_area = 0
		for leaf in leaves:
			if (leaf.is_alive()):
				stem_leaf_area += self.calc_leaf_area(leaf, leaf_conversion)

		total_area = stem_leaf_area * measurement.get_stem_count()
		#LAI = total_area / self._location.get_plant_area() #TODO
		return total_area
		

	def calc_leaf_area(self, leaf, leaf_conversion):
		return (leaf.get_width() * leaf.get_length() * leaf_conversion)
	

	def get_weekly_dd(self, pheno_dates, met_data, fll_date):
		weekly_dd = []
		#prev_date = None

		#first leaf with ligule date - dd is set to 0
		weekly_dd.append({'Date': fll_date, 'dd': 0.0, 'delta_dd': 0.0})

		for pheno_date in pheno_dates:
			delta = timedelta(days = 1)
			current_date = weekly_dd[-1]['Date'] + delta
			current_dd = weekly_dd[-1]['dd']
			cumul_dd = 0.0 
			while current_date <= pheno_date:
				reading = next(x for x in met_data
									if x['Date'] == current_date)

				delta_dd = degree_days(reading['t_max'], reading['t_min'], 0.0)

				if (delta_dd):
					current_dd += delta_dd
					cumul_dd += delta_dd

				current_date += delta

			weekly_dd.append({'Date': pheno_date, 'dd': current_dd, 'delta_dd': cumul_dd})

		return weekly_dd

	def calc_average_LER(self, LER):
		avg_LER_list = [] 
		genotypes = set([x['Genotype'] for x in LER])
		for genotype in genotypes:
			geno_LER = [x for x in LER if x['Genotype'] == genotype]
			stages = set([x['stage'] for x in geno_LER])

			for stage in stages:
				data_points = [x for x in geno_LER if x['stage'] == stage]
				LER_vals = [x['LER'] for x in data_points]
				LER_avg = sum(LER_vals)/len(LER_vals)
				sem = stats.sem(LER_vals)
				if data_points[0]['psi']:
					psi_vals = [x['psi'] for x in data_points]
					psi_avg = sum(psi_vals)/len(psi_vals)
				else:
					psi_avg = False

				avg_LER_list.append({'Genotype': genotype,
										'stage': stage,
										'LER' : LER_avg,
										'sem' : sem,
										'psi' : psi_avg})

		return avg_LER_list

	def get_average_LER(self):
		return self._avg_LER
