#Abstract class that defines the PhenoReader interface - 
#all PhenoReader classes should be implementing the functions here

from abc import ABCMeta, abstractmethod

class MetDataReader(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def __init__(self, location, t_base):
		pass

	def get_records(self):
		return self._met_data
