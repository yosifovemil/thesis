# A gene used in GA_pheno_model
import random


class PhenoGene:
    def __init__(self, name, gene_type, cost, value=None):
        self._name = str(name)
        self._type = gene_type
        self._cost = cost

        if value is None:
            # generate gene
            self._value = bool(random.getrandbits(1))
        else:
            self._value = value

    def get_name(self):
        return self._name

    def get_type(self):
        return self._type

    def get_cost(self):
        return self._cost

    def get_value(self):
        return self._value

    def same(self, gene):
        if self._name == gene.get_name() and \
           self._type == gene.get_type() and \
           self._cost == gene.get_cost() and \
           self._value == gene.get_value():
            return True
        else:
            return False

    def __repr__(self):
        return "%s %s %f %s" % (self._name, self._type,
                                self._cost, str(self._value))
