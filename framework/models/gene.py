# A gene for the GA processing model

from framework.models.naive_ml_process_model import NaiveMLProcessModel
from framework.models.ml_model import MLModel


class Gene:
    _VARIABLE = 'variable'
    _METHOD = 'method'
    _TYPES = [_VARIABLE, _METHOD]
    _KEYS = [''] + \
        NaiveMLProcessModel._STAGE_ONE + \
        NaiveMLProcessModel._STAGE_TWO
    _VARIABLE_VALUES = range(6)
    _METHOD_VALUES = MLModel._METHODS

    def __init__(self, gene_type, key, value):
        self._gene_type = gene_type
        self._key = key
        self._value = value

    def get_type(self):
        return self._gene_type

    def get_key(self):
        return self._key

    def get_value(self):
        return self._value

    def same(self, gene):
        condition = (self._gene_type == gene.get_type()) and \
                    (self._key == gene.get_key()) and \
                    (self._value == gene.get_value())

        return condition
