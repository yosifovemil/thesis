class Plot:
	def __init__(self, UID, genotype):
		self._UID = UID
		self._measurements = [] 
		self._mean_measurements = [] 
		self._genotype = genotype

	def add_measurement(self, plant_UID, LAI, dd, date):
		measurement = {'plant_UID': plant_UID, 
						'LAI': LAI, 'dd': dd, 
						'Date': date}
		self._measurements.append(measurement)	
	
	def get_measurements(self):
		return self._measurements
	
	def get_UID(self):
		return self._UID
	
	def get_genotype(self):
		return self._genotype
	
	def set_plateau(self, plateau):
		self._plateau = plateau.get_plateau()
	
	def get_mean_measurements(self, refresh = False):
		if not self._mean_measurements or refresh:
			self._mean_measurements = [] 
			#mean_measurements is empty - we have to do the math
		 	dates = set([x['Date'] for x in self._measurements])
			for date in dates:
				plants = [x for x in self._measurements if x['Date'] == date]
				cumul_LAI = 0
				for plant in plants:
					cumul_LAI += plant['LAI']
				
				mean_LAI = cumul_LAI / len(plants)
				
				#dd is on plot level - no need to take the mean
				dd = plants[0]['dd'] 
				
				self._mean_measurements.append({'Date': date, 
												'LAI': mean_LAI,
												'dd': dd})

			self._mean_measurements.sort(key = lambda x: x['Date'])
		
		return self._mean_measurements
				
	def set_measurements(self, measurements):
		self._measurements = measurements
