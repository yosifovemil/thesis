import itertools
from datetime import datetime
from framework.models.plateau_detector import PlateauDetector

class LAIPlateauDetector:

	def __init__(self, plants, location, genotype):
		leaf_conversion= location.get_leaf_conversion(genotype)
		#get the mean LAI per plots 
		leaf_areas = self.mean_plots(plants, leaf_conversion)
		
		#mean between plots
		mean_LAI = self.mean_genotype(leaf_areas)
		
		#convert to plateu detector friendly x and y
		mean_LAI.sort(key = lambda x: x['Date'])
		for entry in mean_LAI:
			day = int(entry['Date'].strftime("%j"))
			entry['x'] = day
			entry['y'] = entry['LAI']
	
		detector = PlateauDetector(mean_LAI)

		self._plateau = datetime.strptime("%d %d" % (detector.get_plateau(), 
											location.get_year()), "%j %Y")
		
	def get_plateau(self):
		return self._plateau


	def mean_genotype(self, leaf_areas):
		result = []
		dates = set([x['Date'] for x in leaf_areas])
		for date in dates:
			today_LAI = [x['LAI'] for x in leaf_areas if x['Date'] == date]
			mean_LAI = sum(today_LAI)/len(today_LAI)
			result.append({'Date': date, 'LAI': mean_LAI})

		return result

	def mean_plots(self, plants, leaf_conversion):
		plot_uids = set([x.get_plot() for x in plants])
		plot_leaf_areas = []
		for plot_uid in plot_uids:
			plot_plants = [x for x in plants if x.get_plot() == plot_uid]
			measurements = [x.get_pheno_measurements() for x in plot_plants]
			measurements = list(itertools.chain(*measurements))
			dates = set([x.get_date() for x in measurements])
			for date in dates:
				#get all measurements for this date
				measurements_today = [x for x in measurements 
										if x.get_date() == date]

				#get leaf areas per measurement
				leaf_areas = []
				for measurement in measurements_today:
					leaf_area_per_stem = sum([leaf.get_width() * leaf.get_length() * 
								leaf_conversion for leaf in measurement.get_leaves()])
					leaf_area = leaf_area_per_stem * measurement.get_stem_count()

					leaf_areas.append(leaf_area)
				
				if not leaf_areas:
					import ipdb; ipdb.set_trace()

				mean_leaf_area = sum(leaf_areas)/len(leaf_areas)
				plot_leaf_areas.append({'Plot': plot_uid, 
										'LAI': mean_leaf_area,	
										'Date': date})

		return plot_leaf_areas
