import numpy as np
from scipy.optimize import curve_fit

from framework.models.lm import LM
import copy

class PlateauDetector:
	def __init__(self, data, threshold = None):
		self._data = sorted(data, key = lambda k: k['x'])
		if threshold == None:
			self._threshold = self.find_min_threshold()
		else:
			self._threshold = threshold

		self._plateau = self.perform_magic()
	
	def perform_magic(self):
		"""
		Identifies the plateau in the dataset
		"""
		while len(self._data):
			x,y = self.perform_more_magic(self._data)

			mod = LM(x,y)
			if mod._beta != None and mod._beta <= self._threshold:
				return self._data[0]['x']
	
			del self._data[0]

		return None

	def perform_more_magic(self, data):
		"""
		Converts list of dicts {x,y} to x and y lists
		"""
		x = [data_point['x'] for data_point in data]
		y = [data_point['y'] for data_point in data]
		
		return (x,y)
	
	def get_plateau(self):
		return self._plateau

	def find_min_threshold(self):
		temp_data = copy.deepcopy(self._data)
		min_threshold = 1000
		while len(temp_data):
			x,y = self.perform_more_magic(temp_data)
			mod = LM(x,y)
			if mod._beta < min_threshold and mod._beta != None:
				min_threshold = mod._beta

			del temp_data[0]

		return min_threshold
			
