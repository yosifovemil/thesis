#Class that implements the bootstrap method - random sampling with replacement from a given dataset

import random

class Bootstrap:
	def __init__(self, data, key):
		self._data = [x for x in data if x[key] != None]
		if len(self._data) == 0:
			raise EmptyDataException()

	def generate_subsets(self, n_subsets, n_elements = None):
		if n_elements == None:
			self._subset_size = len(self._data)
		else:
			self._subset_size = n_elements

		subsets = []
		for i in range(n_subsets):
			subset = []

			for j in range(self._subset_size):
				subset.append(random.choice(self._data))

			subsets.append(subset)

		for sub in subsets:
			if len(subset) <= 1:
				raise EmptySubsetsException()

		return subsets

	def get_n(self):
		return self._subset_size

class EmptyDataException(Exception):
	pass

class EmptySubsetsException(Exception):
	pass
