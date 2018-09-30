#Utility class that facilitates csv reading

import csv

class CSVFileReader():
	def __init__(self, filename):
		self._filename = filename
		self.process_input()
	
	def process_input(self):
		f = open(self._filename, 'r')
		self._content = []
		reader = csv.DictReader(f)
		for row in reader:
			self._content.append(row)

		f.close() 
	
	def get_content(self):
		return self._content
