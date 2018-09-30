class Plant:
	def __init__(self, UID, genotype):
		#sanity check - UID is string
		if type(UID) != str:
			raise Exception("Plant UID needs to be of str type")

		self._UID = UID
		self._genotype = genotype
		self._pheno_measurements = []
		self._sunscan_measurements = [] 
		self._DMY = 0

	def get_uid(self):
		return self._UID

	def get_genotype(self):
		return self._genotype

	def add_phenomeasurement(self, measurement):
		if self._pheno_measurements:
			prev_measurement = self._pheno_measurements[-1]
			if (measurement.get_date() < prev_measurement.get_date()):
				raise Exception("Attempt to add measurement from earlier date")

		self._pheno_measurements.append(measurement)

	def add_sunscanmeasurement(self, measurement):
		if self._sunscan_measurements:
			#first see if we are overwriting a previous measurement
			temp_uid = measurement._temp_uid
			current_date = measurement.get_date()

			#get all the previous repeat measurements if any
			match = [index for index,x in enumerate(self._sunscan_measurements)
							if x._temp_uid == temp_uid and
								x.get_date() == current_date]
	
			if (len(match) == 1):
				#make sure we do not overwrite with older measurement
				prev_measurement = self._sunscan_measurement[match[0]]
				if (prev_measurement.get_datetime() > measurement.get_datetime()):
					raise Exception("Attempt to overwrite measurement "
										"with one from earlier date")

				#overwrite the measurement
				self._sunscan_measurements[match[0]] = measurement
				return 

			elif(len(match) > 1):
				raise Exception("This is a cosmic scale exception")

			#check if we are adding a measurement from a previous date, i.e.
			prev_measurement = self._sunscan_measurements[-1]
			if (measurement.get_date() < prev_measurement.get_date()):
				raise Exception("Attempt to add measurement from earlier date")

		self._sunscan_measurements.append(measurement)

	def set_DMY(self, DMY):
		self._DMY = DMY

	def get_last_phenomeasurement(self):
		return self._pheno_measurements[-1]
	
	def get_pheno_measurements(self):
		return self._pheno_measurements

	def get_sunscan_measurements(self):
		return self._sunscan_measurements

	def get_plot(self):
		return self._UID.split("/")[0]
