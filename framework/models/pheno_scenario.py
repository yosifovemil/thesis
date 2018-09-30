# An individual scenario used for the GA_pheno_model
from framework.models.pheno_gene import PhenoGene
import framework.util.config_parser as config_parser
import os
import framework.data.formats as formats
from copy import deepcopy
import random
from framework.util.csv_io.csv_filewriter import CSVFileWriter
import logging


class PhenoScenario:
    def __init__(self, genes=None, **kwargs):
        """
        scenario_type - possible values ['pheno', 'var', 'pheno_var']

        kwargs - ['week'] = names of weeks
                 ['variable'] = names of variables
        """

        self._kwargs = kwargs
        self._config = config_parser.SpaceAwareConfigParser()
        self._config.read(os.path.expandvars("$ALF_CONFIG/pheno_scenario.ini"))

        if genes is not None:
            self._genes = genes
        else:
            self._genes = self._generate_genes()

        self._genes_len = len(self._genes)

        self._max_cost = self._calculate_max_cost()
        self._cost = self._calculate_cost()

    def cross(self, pheno_strategy):
        random.seed()
        new_genes = []
        max_crosses = 2
        crosses_n = random.randint(1, max_crosses)

        assert(len(self._genes) == len(pheno_strategy.get_genes()))

        cross_points = []
        while len(cross_points) < crosses_n:
            cross_p = random.randint(1, len(self._genes) - 2)
            if cross_p not in cross_points:
                cross_points.append(cross_p)

        cross_points.sort()

        new_genes = self._genes[0:cross_points[0]]
        if crosses_n == 1:
            new_genes += pheno_strategy.get_genes()[cross_points[0]:]
        else:
            new_genes += pheno_strategy.get_genes()[cross_points[0]:
                                                    cross_points[1]]
            new_genes += self._genes[cross_points[1]:]

        assert(len(new_genes) == self._genes_len)

        return PhenoScenario(genes=new_genes, **self._kwargs)

    def viable(self, data):
        key = 'ml_data'
        all_years = sorted(list(set([x[formats.DATE].year for x in
                                     data[key]
                                     if x[formats.DW_PLANT] is not None])))
        data_sub = self.apply(data)
        sub_years = sorted(list(set([x[formats.DATE].year for x in
                                     data_sub[key]
                                     if x[formats.DW_PLANT] is not None])))

        if all_years != sub_years:
            return False

        for year in all_years:
            year_data = [x for x in data_sub[key]
                         if x[formats.DATE].year == year and
                         x[formats.DW_PLANT] is not None]
            uniq_values = set([x[formats.DW_PLANT] for x in year_data])
            if len(uniq_values) < 6:
                logging.warn("Weird pheno strategy:")
                logging.warn("Year: %d, key %s" % (year, key))
                logging.warn(self.__repr__())
                return False

        return True

    def apply(self, data):
        new_data = dict()
        for key in data:
            subset = deepcopy(data[key])
            for gene in self._genes:
                if gene.get_value() is False:
                    if gene.get_type() == 'week':
                        subset = [x for x in subset if
                                  x[formats.DATE].strftime("%W") !=
                                  gene.get_name()]
                    elif gene.get_type() == 'variable' and key == 'ml_data':
                        # only delete the ml_data vars, do not touch
                        # them for Chris Davey's model
                        for entry in subset:
                            del entry[gene.get_name()]

            new_data[key] = subset

        return new_data

    def same(self, scenario):
        scenario_genes = scenario.get_genes()
        same = True
        for i in range(len(self._genes)):
            if not self._genes[i].same(scenario_genes[i]):
                same = False
                break

        return same

    def get_genes(self):
        return self._genes

    def get_cost(self):
        return self._cost / self._max_cost

    def _generate_genes(self):
        genes = []
        for key in self._kwargs.keys():
            for entry in self._kwargs[key]:
                if key == 'week':
                    cost = -999
                elif key == "variable":
                    cost = float(self._config.get('cost', entry))
                else:
                    raise Exception("You are exceptional and clueless!")

                gene = PhenoGene(entry, key, cost)
                genes.append(gene)

        return genes

    def _calculate_cost(self):
        var_genes = [x for x in self._genes if x.get_type() == 'variable'
                     if x.get_value()]
        var_cost = sum([x.get_cost() for x in var_genes])

        weeks = [x for x in self._genes if x.get_type() == 'week'
                 if x.get_value()]

        cost = var_cost * len(weeks)
        return cost

    def _calculate_max_cost(self):
        var_genes = [x for x in self._genes if x.get_type() == 'variable']
        var_cost = sum([x.get_cost() for x in var_genes])

        weeks = [x for x in self._genes if x.get_type() == 'week']

        cost = var_cost * len(weeks)
        return cost

    def __repr__(self):
        matrix = []
        for gene in self._genes:
            matrix.append([gene.get_name(), gene.get_type(),
                           gene.get_cost(), str(gene.get_value())])

        header = ['name', 'type', 'cost', 'value']

        result = self._matrix_to_string(matrix, header)
        result += "\nCost %f\n" % self.get_cost()
        return result

    def _matrix_to_string(self, matrix, header=None):
        """
        Return a pretty, aligned string representation of a nxm matrix.

        This representation can be used to print any tabular data, such as
        database results. It works by scanning the lengths of each element
        in each column, and determining the format string dynamically.

        @param matrix: Matrix representation (list with n rows of m elements).
        @param header: Optional tuple or list with header elements to be
        displayed.
        """
        if type(header) is list:
            header = tuple(header)
        lengths = []
        if header:
            for column in header:
                lengths.append(len(column))
        for row in matrix:
            for column in row:
                i = row.index(column)
                column = str(column)
                cl = len(column)
                try:
                    ml = lengths[i]
                    if cl > ml:
                        lengths[i] = cl
                except IndexError:
                    lengths.append(cl)

        lengths = tuple(lengths)
        format_string = ""
        for length in lengths:
            format_string += "%-" + str(length) + "s "
        format_string += "\n"

        matrix_str = ""
        if header:
            matrix_str += format_string % header
        for row in matrix:
            matrix_str += format_string % tuple(row)

        return matrix_str

    def to_csv(self, fname):
        header = ['name', 'type', 'value']
        content = []
        for gene in self._genes:
            content.append({'name': gene.get_name(),
                            'type': gene.get_type(),
                            'value': str(gene.get_value())})

        CSVFileWriter(fname, content, header)
