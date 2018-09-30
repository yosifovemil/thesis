#Reader of transmissions values for BSBEC 2011 and 2012 data

from framework.util.csv_io.csv_filereader import CSVFileReader

import os
from datetime import datetime

class TransmissionReader:
	def __init__(self, location):
		self._location = location
		#check whether transmissions data file exists
		self._file_location = location.get_transmission_location()
		if not os.path.exists(self._file_location):
			raise Exception("No transmission data file found!")

	def get_data(self):
		transmission_data = CSVFileReader(self._file_location).get_content()
		for entry in transmission_data:
			entry['Date'] = datetime.strptime("%s %s" % (entry['Day'], entry['Year']),
												"%j %Y")
			entry['Day'] = int(entry['Day'])
			entry['Transmission'] = float(entry['Transmission'])

		return transmission_data
