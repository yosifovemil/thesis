# A GA based algorithm to predict importance of variables and weeks in data
import framework.data.formats as formats
from framework.models.GA_pheno_model import GAPhenoModel
from framework.models.ml_process_model import MLProcessModel
from framework.models.naive_ml_process_model import NaiveMLProcessModel
import logging
from framework.util.csv_io.csv_filewriter import CSVFileWriter
from framework.util.misc import cdf
import random
from datetime import datetime
import os
import gc
import psutil
import copy


class GAPhenoAlgorithm:
    def __init__(self, data, variables, population_size, model_names,
                 root_dir, max_cost, scenario):
        """
        data - full dataset
        variables - variables allowed to be used/dropped from models
        population_size - size of the population
        model_names - models to be used for evaluating each solution
        report_file - file to report best individual from each generation
        """
        random.seed()

        for key in data:
            temp_data = [x for x in data[key] if
                         x[formats.DATE].month > 3 and
                         x[formats.DATE].month < 11]
            data[key] = temp_data

        self._data = data
        self._variables = variables

        # fix missing data points
        ml_data = []

        desired_vars = copy.deepcopy(self._variables)
        if scenario == 'simple_ml':
            desired_vars.append(formats.DW_PLANT)
        elif scenario == 'process_ml':
            pass
        else:
            raise Exception("Scenario %s is magnifico" % scenario)

        for entry in self._data['ml_data']:
            add = True
            for key in desired_vars:
                if entry[key] is None:
                    add = False

            if add:
                ml_data.append(entry)

        self._data['ml_data'] = ml_data

        self._population = []
        self._population_size = population_size
        self._model_names = model_names
        self._max_cost = max_cost
        self._max_cost_str = "{:.2f}".format(self._max_cost)
        self._stop_file = os.path.join(os.environ['HOME'],
                                       '.ga_signals',
                                       self._max_cost_str)

        self._best_models = []  # to be populated on each generation

        # model parameters
        self._keep_rate = 0.2
        self._cross_rate = 0.6
        self._immigration_rate = 0.2
        self._mutation_rate = 0.2

        if self._population_size == 10:
            self._probabilities = cdf(0.2, 1.2, population_size)  # for 10
        elif self._population_size == 25:
            self._probabilities = cdf(0.2, 1.2485, population_size)
        elif self._population_size == 50:
            self._probabilities = cdf(0.2, 1.249995, population_size)  # for 50
        else:
            import ipdb
            ipdb.set_trace()

        # set up report file
        self._root_dir = root_dir
        self._report_file = os.path.join(root_dir, 'report.csv')
        self._report_header = ['generation', 'pheno_cost']
        self._report_header += model_names
        self._report_header.append('total')

        self._ram_file = os.path.join(root_dir, 'ram.csv')
        CSVFileWriter(self._ram_file, [], ['date', 'ram'])

        CSVFileWriter(self._report_file, [], self._report_header)

        logging.info("Calculating max RMSEs")
        self._max_rmse = self._get_max_rmse()
        self.run_model()

    def run_model(self):
        durations = []
        i = 1
        stop = False
        logging.info("[%s]Generating first generation" % self._max_cost_str)
        self._population = self._generate_individuals([],
                                                      self._population_size)
        while not stop:
            logging.info("[%s]Generation %d started" % (self._max_cost_str, i))
            start_time = datetime.now()
            self._population = self.run_generation(i)

            # manage duration time
            duration = (datetime.now() - start_time).seconds
            durations.append(duration)
            mean_duration = float(sum(durations))/len(durations)

            # report
            logging.info("[%s]Generation %d finished" % (self._max_cost_str,
                                                         i))
            logging.info("[%s]Generation %d lasted %d seconds" %
                         (self._max_cost_str, i, duration))
            logging.info("[%s]Mean generation time %d seconds" %
                         (self._max_cost_str, mean_duration))
            i += 1

            gc.collect()
            # check if we need to stop
            stop = self._check_stop()

        logging.info("[%s]Stopping condition met, stopping GA..." %
                     self._max_cost_str)

    def run_generation(self, generation_n):
        # report on this generation
        self._population.sort(key=lambda x: x.get_value())
        self._report(self._population, generation_n)
        self._best_models.append(self._population[0])

        keep = self._population[0:int(self._keep_rate * self._population_size)]

        logging.info("[%s]Crossing population" % self._max_cost_str)
        crosses = self._perform_crosses(keep, self._population)

        logging.info("[%s]Generating immigrants" % self._max_cost_str)
        immigrants = self._generate_individuals(keep + crosses,
                                                self._immigration_rate *
                                                self._population_size)

        return keep + crosses + immigrants

    def _perform_crosses(self, keep, population):
        crosses = []
        while len(crosses) < self._cross_rate * self._population_size:
            probs = [random.random(), random.random()]
            parents = []
            for prob in probs:
                index = self._probabilities.index(next(x for x in
                                                       self._probabilities
                                                       if x > prob))
                individual = population[index]
                parents.append(individual)

            assert(len(parents) == 2)

            new_cross = parents[0].cross(parents[1])

            skip = False
            for entry in crosses + keep:
                if entry.same(new_cross):
                    skip = True

            if not skip:
                crosses.append(new_cross)

        return crosses

    def _generate_individuals(self, population, n):
        immigrants = []
        while len(immigrants) < n:
            immigrant = GAPhenoModel(self._data, self._variables,
                                     self._model_names, self._max_rmse,
                                     self._max_cost)

            skip = False
            for entry in population + immigrants:
                if entry.same(immigrant):
                    skip = True

            if not skip:
                immigrants.append(immigrant)

        return immigrants

    def _get_max_rmse(self):
        results = dict()

        all_vars = NaiveMLProcessModel._INIT_DATA + \
            NaiveMLProcessModel._STAGE_ONE

        for method in self._model_names:
            model = MLProcessModel(method, self._data,
                                   self._data, all_vars)

            results[method] = model._rmse_abs

            logging.info("[%s]MAX RMSE %s: %f" % (self._max_cost, method,
                                                  model._rmse_abs))

        return results

    def _report(self, population, generation):
        best = population[0]
        rep = "[%s]Generation %d\n" % (self._max_cost_str, generation)
        rep += "[%s]Best solution:\n" % self._max_cost_str
        rep += "[%s]Pheno cost: %f\n" % (self._max_cost_str,
                                         best._pheno_scenario.get_cost())

        rep += "[%s]%s" % (self._max_cost_str,
                           str([x.get_value() for x in population]))

        logging.info(rep)

        entry = {'generation': generation,
                 'pheno_cost': best._pheno_scenario.get_cost()}

        for model_name in self._model_names:
            models = []
            for rep in best.get_reps():
                models += [x for x in rep if x.get_method() == model_name]

            val = sum([x.get_rmse() for x in models])/len(models)
            entry[model_name] = val

        entry['total'] = best.get_value()

        CSVFileWriter(self._report_file, [entry], self._report_header, 'a')

        meta = []
        for individual in population:
            fname = os.path.join(self._root_dir,
                                 "%d.csv" % population.index(individual))

            individual.get_pheno_scenario().to_csv(fname)
            for rep, rep_value in zip(individual.get_reps(),
                                      individual.get_rep_values()):
                meta_entry = {'name': fname,
                              'value': individual.calc_value(),
                              'cost':
                              individual.get_pheno_scenario().get_cost(),
                              'rep': individual.get_reps().index(rep) + 1}

                for model in rep:
                    meta_entry[model.get_method()] = model.get_rmse()

                meta.append(meta_entry)

        CSVFileWriter(os.path.join(self._root_dir, "meta.csv"), meta)

    def _check_stop(self):
        # check for manual requests
        if os.path.exists(self._stop_file):
            return True

        if self._check_memory():
            logging.info("Memory full, stopping...")
            return True

        # don't bother if below generation 150
        if len(self._best_models) < 150:
            return False

        # last 100 models need to have been the same
        last_model = self._best_models[-1]
        for i in reversed(range(-100, 0)):
            if not last_model.same(self._best_models[i]):
                return False

        return True

    def _check_memory(self):
        p = psutil.Process(os.getpid())
        ram = p.memory_full_info()[7]
        logging.info("RAM: %s" % self.bytes2human(ram))

        entry = {'date': datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                 'ram': ram}
        CSVFileWriter(self._ram_file, [entry], ['date', 'ram'])

        # larger than 20G
        if ram >= 21474836480:
            return True
        else:
            return False

    def bytes2human(self, n):
        # http://code.activestate.com/recipes/578019
        # >>> bytes2human(10000)
        # '9.8K'
        # >>> bytes2human(100001221)
        # '95.4M'
        symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
        prefix = {}
        for i, s in enumerate(symbols):
            prefix[s] = 1 << (i + 1) * 10
        for s in reversed(symbols):
            if n >= prefix[s]:
                value = float(n) / prefix[s]
                return '%.1f%s' % (value, s)
        return "%sB" % n
