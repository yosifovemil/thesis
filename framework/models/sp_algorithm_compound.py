# ScatterPhenoAlgorithm - the algorithm class used for running scatter search
# for the best phenotyping scenario
import framework.data.formats as formats
from copy import deepcopy
from rpy2.robjects.packages import importr
from framework.models.scatter_pheno_model import ScatterPhenoModel
from framework.models.sp_scenario_builder \
    import SPScenarioBuilder, NoValidSolutionException
from framework.models.scatter_pheno_scenario_container \
        import ScatterPhenoScenarioContainer, \
        ScatterPhenoScenarioContainerException
from framework.models.scatter_pheno_scenario import ScatterPhenoScenario
from framework.models.ml_process_model import MLProcessModel
from pathos.multiprocessing import ProcessingPool as Pool
from framework.util.csv_io.csv_filewriter import CSVFileWriter
from framework.util.csv_io.csv_filereader import CSVFileReader
import os
from datetime import datetime
import logging
from framework.util.resource_tools import memory, total_memory, res_memory
from framework.util.email_send import send_email


glob_base = importr("base")


class ScatterPhenoAlgorithm:

    def __init__(self, data, scenario, root_dir, load=False):
        """data - {'cd_data': [], 'ml_data': []}
        scenario - 'simple_ml' or 'process_ml'
        """

        # hardcoded algorithm variables, could supply them to the
        # constructor if needed
        # self._PSize = 45 TODO real value
        self._PSize = 12

        # weight for previous score entry, when updating the score table
        self._alpha = 0.3

        # self._b = 20 TODO real value
        self._b = 8

        self._proc_count = 4

        # set class variables
        self._variables = [formats.STEM_COUNT, formats.CANOPY_HEIGHT,
                           formats.TRANSMISSION, formats.FLOWERING_SCORE,
                           formats.LEAF_AREA_INDEX, formats.COL,
                           formats.ROW, formats.DD, formats.GENOTYPE,
                           formats.RAINFALL, formats.DOY, formats.PAR]
        self._variables.sort()

        self._scenario = scenario
        self._root_dir = root_dir
        if scenario == "simple_ml":
            self._methods = ['rf', 'knn', 'gbm']
        elif scenario == "compound":
            self._methods = ['NaiveMLProcessModelMemfix', 'GAWinModel']
        else:
            raise Exception("STUB")  # TODO

        self._data = self._hack_data(data)

        self._months = list(set([x[formats.DATE].strftime("%B") for x in
                                self._data['ml_data']]))
        self._months.sort()

        # find maximum RMSE for methods
        self._max_rmse = self._get_max_rmse()

        # DB to contain all solutions ever explored
        self._database = ScatterPhenoScenarioContainer()
        self._score_table = self._empty_score_table()
        if load:
            sc_file = os.path.join(self._root_dir, 'score_table.csv')
            self._score_table = CSVFileReader(sc_file).get_content()
            for entry in self._score_table:
                entry['score'] = float(entry['score'])
                entry['value'] = (entry['value'] == "True")

            db_file = os.path.join(self._root_dir, 'database.csv')
            self._database.load_file(db_file, self._data)
            self._update_score_table()

            self._run_algorithm2()
        else:
            self._run_algorithm()

    def _run_algorithm2(self):
        """Algorithm main method"""
        scenario_builder = SPScenarioBuilder(self._data,
                                             self._variables,
                                             process=True)
        scenario_builder.load_cm(os.path.join(self._root_dir,
                                 'cm_functions.csv'))

        # G1
        logging.info("G1")
        population = scenario_builder.g1(self._PSize/3)
        self._update_score_table(population)

        # G2
        logging.info("G2")
        population = self._generator(population, scenario_builder.g2)

        # G3
        logging.info("G3")
        population = self._generator(population, scenario_builder.g3)

        # form ref set from database
        ref_set = self._ref_set_update(self._database)

        self._report(ref_set, scenario_builder)
        # Leaving this here in case I change my mind TODO
        # ref_set = ScatterPhenoScenarioContainer()
        # ref_set.load_file(os.path.join(self._root_dir, 'ref_set.csv'),
        #                   self._data)

        self._main_loop(ref_set, scenario_builder, population)

    def _run_algorithm(self):
        """Algorithm main method"""
        scenario_builder = SPScenarioBuilder(self._data,
                                             self._variables,
                                             process=True)

        # G1
        logging.info("G1")
        start = datetime.now()
        population = scenario_builder.g1(self._PSize/3)
        self._update_score_table(population)
        logging.info("G1 - %s" % (datetime.now() - start))

        # G2
        logging.info("G2")
        start = datetime.now()
        population = self._generator(population, scenario_builder.g2)
        logging.info("G2 - %s" % (datetime.now() - start))

        # G3
        start = datetime.now()
        logging.info("G3")
        population = self._generator(population, scenario_builder.g3)
        logging.info("G3 - %s" % (datetime.now() - start))

        # build ref set
        logging.info("Building ref_set")
        ref_set = self._ref_set_update(population)

        self._report(ref_set, scenario_builder)

        # parallel improvement of the best b/2 solutions
        start = datetime.now()
        logging.info("Improving ref_set")
        ref_set = self._mp_improve(ref_set, scenario_builder)
        logging.info("Improvements - %s" % (datetime.now() - start))

        self._main_loop(ref_set, scenario_builder, population)

    def _main_loop(self, ref_set, scenario_builder, population):
        stop = False
        last_changed = 0
        iteration = 0
        while not stop:
            start_loop = datetime.now()
            # create the pool from combining solutions from ref_set
            logging.info("Performing combinations")
            start = datetime.now()
            pool = self._combine(ref_set, scenario_builder)
            logging.info("Combinations %s" % (datetime.now() - start))

            # improve pool
            logging.info("Improving best combinations")
            start = datetime.now()
            pool = self._mp_improve(pool, scenario_builder)
            logging.info("Improvements %s" % (datetime.now() - start))

            # join ref_set and pool together
            union = deepcopy(ref_set)
            union.add_container(pool)
            union.sort()

            new_ref_set = self._ref_set_update(union)
            if ref_set.same(new_ref_set):
                logging.info("Ref_set not changed")
                new_ref_set = ScatterPhenoScenarioContainer()

                for i in range(self._b/2):
                    new_ref_set.add(union.get(i))

                # get the most diverse solutions to what
                # we already have in ref_set
                while new_ref_set.len() < self._b:
                    new_ref_set.add(population.get_diverse(new_ref_set))

            if ref_set.same(new_ref_set):
                last_changed += 1
            else:
                last_changed = 0

            if last_changed >= 5:
                logging.info("Reached optimal solution, terminating...")
                stop = True

            if os.path.exists('/home/eey9/.stop_scatter_search'):
                stop = True
                logging.info("Stopping because of file flag...")

            ref_set = new_ref_set
            iteration += 1
            logging.info("Completed iteration %d" % iteration)
            self._report(ref_set, scenario_builder)
            t_delta = datetime.now() - start_loop
            logging.info("Iteration time %s" % t_delta)

        return

    def _ref_set_update(self, source):
        source.sort()
        ref_set = ScatterPhenoScenarioContainer()
        for i in range(self._b/2):
            ref_set.add(source.get(i))

        # get the most diverse solutions to what we already have in ref_set
        while ref_set.len() < self._b:
            ref_set.add(source.get_diverse(ref_set))

        return ref_set

    def _combine(self, container, scenario_builder):
        # build subsets
        combinations = self._build_combinations(container)
        pool = ScatterPhenoScenarioContainer()
        for combination in combinations:
            start = datetime.now()
            try:
                new_scenario = scenario_builder.combine(combination[0],
                                                        combination[1],
                                                        self._score_table)
            except NoValidSolutionException:
                logging.info("Combination %d/%d - %s: no valid solution" %
                             (combinations.index(combination) + 1,
                              len(combinations),
                              (datetime.now() - start)))
                continue

            self._update_score_table(new_scenario)
            if not pool.contains(new_scenario):
                pool.add(new_scenario)

            # see where does the scenario qualify to be in container
            try:
                j = container.index(next(x for x in container.get_all()
                                    if new_scenario.get_utility() <
                                    x.get_utility()))
                scenario_builder.success(self._b - j)
            except StopIteration:
                continue  # Worse than anything in ref_set, does not qualify

            logging.info("Combination %d/%d - %s" %
                         (combinations.index(combination) + 1,
                          len(combinations),
                          (datetime.now() - start)))

        return pool

    def _sp_improve(self, container, scenario_builder):
        container.sort()

        best = []
        for i in range(self._b/2):
            best.append(container.get(i))

        result = []
        for scenario in best:
            result.append(self._improve(scenario, scenario_builder))

        for entry in result:
            index = container.index(entry['individual'])
            best = entry['improvements'].get(0)
            if best.get_utility() < entry['individual'].get_utility():
                container.replace(best, index)

            for improvement in entry['improvements'].get_all():
                self._update_score_table(improvement)

        logging.info("Improved %d solutions" % container.get_changes())
        container.reset_changes()
        return container

    def _mp_improve(self, container, scenario_builder):
        """Improves b/2 best solutions from the container and updates
        the score table with the generated solutions
        """
        container.sort()
        pool = Pool(processes=self._proc_count)

        logging.info("Starting processes")
        start = datetime.now()
        best = []
        builders = []
        for i in range(self._b/2):
            best.append(container.get(i))
            builders.append(scenario_builder)

        try:
            result = pool.map(self._improve, best, builders)
            pool.close()
            pool.join()
        except MemoryError as e:
            send_email("I crashed again, please help!")
            import pudb
            pudb.set_trace()
            print(e.message())

        logging.info("Processes finished - %s" % (datetime.now() - start))
        # How infuriating was that?!
        # pathos was being smart and was caching pool so this is needed
        # to prevent from erroring out
        pool.restart()

        start = datetime.now()
        logging.info("mp_improve second loop")
        for entry in result:
            index = container.index(entry['individual'])
            best = entry['improvements'].get(0)
            if best.get_utility() < entry['individual'].get_utility():
                container.replace(best, index)

            for improvement in entry['improvements'].get_all():
                self._update_score_table(improvement)

        logging.info("mp_improve second loop - %s" % (datetime.now() - start))
        logging.info("Improved %d solutions" % container.get_changes())
        container.reset_changes()
        return container

    def _improve(self, individual, scenario_builder):
        start = datetime.now()
        base = importr("base")
        candidate_list = self._build_candidate_list(individual)

        improvements = ScatterPhenoScenarioContainer()

        for var in candidate_list:
            new_scenario = scenario_builder.flip(individual, var)
            if new_scenario.same(individual) or \
                    not new_scenario.valid(process=True):
                continue

            new_scenario = self._evaluate(new_scenario, base)

            if not improvements.contains(new_scenario):
                improvements.add(new_scenario)

        for i in range(len(individual.get_solution())):
            for j in range(i + 1, len(individual.get_solution())):
                new_scenario = scenario_builder.swap(individual, i, j)
                if new_scenario.same(individual) or \
                        not new_scenario.valid(process=True):
                    continue

                new_scenario = self._evaluate(new_scenario, base)
                if not improvements.contains(new_scenario):
                    improvements.add(new_scenario)

                if not self._database.contains(new_scenario):
                    self._database.add(new_scenario)

        improvements.sort()

        logging.info("self._improve finished - %s" %
                     (datetime.now() - start))
        return {'individual': individual,
                'improvements': improvements}

    def _build_candidate_list(self, individual):
        candidate_list = []
        for entry in individual.get_solution():
            t = next(x for x in entry.keys() if x != 'value')
            candidate_list.append(next(x for x in self._score_table
                                       if x['type'] == t and
                                       entry[t] == x['name'] and
                                       entry['value'] != x['value']))

        # smallest score = highest probability so DONT CHANGE THIS
        candidate_list.sort(key=lambda x: x['score'])
        return candidate_list

    def _generator(self, population, func):
        generated = 0
        while generated < self._PSize/3:
            worked = False
            while not worked:
                try:
                    individual = func(self._score_table)
                    population.add(individual)
                    self._update_score_table(individual)
                    generated += 1
                    worked = True
                except ScatterPhenoScenarioContainerException:
                    # already exists
                    pass

        return population

    def _evaluate(self, pheno_scenario, base=None):
        if base is None:
            base = glob_base

        if self._database.contains(pheno_scenario):
            return self._database.request(pheno_scenario)

        model = ScatterPhenoModel(self._data,
                                  base,
                                  pheno_scenario,
                                  self._methods,
                                  self._max_rmse)

        util = (0.3 * pheno_scenario.get_cost()) + model.get_rmse()
        pheno_scenario.set_utility(util)
        pheno_scenario.set_rmse(model.get_absolute_rmse())
        return pheno_scenario

    def _update_score_table(self, input_=None):
        if input_ is None:
            self._score_table = self._calc_table(self._database.get_all())

        elif input_.__class__ == ScatterPhenoScenarioContainer:
            # we are given a population - this should happen after G1
            population = input_.get_all()

            for individual in population:
                if not self._database.contains(individual):
                    self._evaluate(individual)
                    self._database.add(individual)
                else:
                    logging.info("Warning, individual in database, weird!")

            # calculate score based on whole database population
            self._score_table = self._calc_table(self._database.get_all())

        elif input_.__class__ == ScatterPhenoScenario:
            individual = input_
            if self._database.contains(individual):
                # individual already in database, update the individual
                # with the rmse and utility
                index = self._database.index(individual)

                # but only if it needs updating
                if not individual.has_utility():
                    self._database.get(index).copy_to(individual)

                return

            if not individual.is_evaluated():
                self._evaluate(individual)

            # create the table at t
            new_table = self._calc_table(self._database.get_all() +
                                         [individual])

            # use the tables at t and t-1 to calculate the smoothed score
            for entry, new_entry in zip(self._score_table, new_table):
                entry['score'] = (self._alpha * entry['score'] +
                                  (1 - self._alpha) * new_entry['score'])

            self._database.add(individual)

    def _calc_table(self, population):
        new_table = []
        # using self._score_table just for template here,
        # none of the values will be copied
        for entry in self._score_table:
            new_entry = deepcopy(entry)
            index = self._score_table.index(entry)/2

            match = [x for x in population
                     if x.get(index) == entry['value']]
            non_match = [x for x in population
                         if x.get(index) != entry['value']]

            # little hack to get around certain variables not having
            # both values
            if len(match) == 0 or len(non_match) == 0:
                score = 0.5
            else:
                util_match = (sum([x.get_utility() for x in match]) /
                              len(match))
                util_non_match = (sum([x.get_utility()
                                       for x in non_match]) /
                                  len(non_match))

                score = util_match / (util_match + util_non_match)

            new_entry['score'] = score
            new_table.append(new_entry)

        return new_table

    def _get_max_rmse(self):
        rmses = dict()
        for method in self._methods:
            model = MLProcessModel(method, self._data, self._data,
                                   self._variables, glob_base)

            rmses[method] = model._rmse_abs

        return rmses

    def _build_combinations(self, container):
        container.sort()
        population = container.get_all()

        combinations = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                combination = [population[i], population[j]]
                combinations.append(combination)

        return combinations

    def _hack_data(self, data):
        # remove winter months that are useless
        for key in data.keys():
            new_data = [x for x in data[key]
                        if x[formats.DATE].month > 3 and
                        x[formats.DATE].month < 11]
            data[key] = new_data

        # remove records with missing values for needed variables
        needed_vars = deepcopy(self._variables)

        ml_data = []
        for entry in data['ml_data']:
            add = True
            for key in needed_vars:
                if entry[key] is None:
                    add = False
                    break

            if add:
                ml_data.append(entry)

        data['ml_data'] = ml_data
        return data

    def _empty_score_table(self):
        score_table = []
        for month in self._months:
            for val in [False, True]:
                entry = dict()
                entry['name'] = month
                entry['type'] = 'month'
                entry['value'] = val
                entry['score'] = 0
                score_table.append(entry)

        for var in self._variables:
            for val in [False, True]:
                entry = dict()
                entry['name'] = var
                entry['type'] = 'variable'
                entry['value'] = val
                entry['score'] = 0
                score_table.append(entry)

        return score_table

    def _report(self, ref_set, sb):
        self._to_csv(os.path.join(self._root_dir, 'ref_set.csv'), ref_set)
        CSVFileWriter(os.path.join(self._root_dir, 'score_table.csv'),
                      self._score_table)
        self._to_csv(os.path.join(self._root_dir, 'database.csv'),
                     self._database)

        # delete the function reference from the dict and save the info to file
        func = deepcopy(sb._cm_functions)
        for f in func:
            del f['function']

        CSVFileWriter(os.path.join(self._root_dir, 'cm_functions.csv'), func)

        # report memory usage
        mem_report = [{'date': datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                       'mem': memory()/float(1024**2),
                       'res_memory': res_memory(),
                       'total_memory': total_memory()}]

        memfile = os.path.join(self._root_dir, 'memfile.csv')
        if not os.path.exists(memfile):
            CSVFileWriter(memfile, mem_report)
        else:
            CSVFileWriter(memfile, mem_report, write_mode='a')

    def _to_csv(self, fname, container):
        data = [x.to_dict() for x in container.get_all()]
        CSVFileWriter(fname, data)
