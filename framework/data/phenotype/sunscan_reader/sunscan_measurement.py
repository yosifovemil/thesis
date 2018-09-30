class SunScanMeasurement:
	def __init__(self, PAR, total, diffuse, spread, date, temp_uid):
		self._PAR = PAR
		self._total = total
		self._diffuse = diffuse
		self._spread = spread
		self._date = date
		self._temp_uid = temp_uid

	def get_temp_uid(self):
		return self._temp_uid

	def get_date(self):
		return self._date.date()

	def get_datetime(self):
		return self._date

	def get_interception(self):
		return 1 - (self._PAR / self._total)
