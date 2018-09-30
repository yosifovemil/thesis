#Abstract class that defines the PhenoReader interface - 
#all PhenoReader classes should be implementing the functions here

from abc import ABCMeta, abstractmethod

class PhenoReader(object):
	__metaclass__ = ABCMeta
	_records = []

	@abstractmethod
	def __init__(self, location, *kwargs):
		pass

	def get_records(self):
		return self._records
