# Build an ML process model using genetic algorithms

from framework.models.naive_ml_process_model import NaiveMLProcessModel
from framework.models.gene import Gene
from framework.models.ml_model import MLModel
from framework.util.misc import split_dataset, cdf
from framework.util.csv_io.csv_filewriter import CSVFileWriter
from framework.util.csv_io.csv_filereader import CSVFileReader
from framework.models.GA_custom_model import GACustomModel
import random
from itertools import combinations
import logging
import pickle
import os
from multiprocessing import Process, Lock
from multiprocessing.queues import Queue
from tempfile import mkstemp
from datetime import datetime


class GAProcessModel:
    def __init__(self, data, years, population_size, root_dir,
                 train_submodels=False,
                 process_count=4):
        self._data = data
        self._years = years
        self._population_size = population_size
        self._root_dir = root_dir

        # GA parameters
        self._mutation_rate = 0.2
        self._probabilities = cdf(0.05, 1.05228, population_size)  # for 100
        # self._probabilities = cdf(0.1, 1.1104, population_size)  # for 50
        # self._probabilities = cdf(0.2, 1.246, population_size)  # for 20

        # population proportions
        self._immigration_rate = 0.2
        self._cross_rate = 0.6
        self._keep_rate = 0.2
        self._min_delta = 1.0 / self._population_size

        # min/max population proportions
        self._immigration_rate_max = 0.7
        self._immigration_rate_min = 0.2

        self._cross_rate_min = 0.2
        self._cross_rate_max = 0.6

        self._keep_rate_min = 0.1
        self._keep_rate_max = 0.2

        self._mutation_rate_max = 0.4
        self._mutation_rate_min = 0.2

        self._RMSE_history = []

        self._model_cache = os.path.join(os.environ['HOME'], '.ga_models')
        self._process_count = process_count
        random.seed()

        if train_submodels:
            # remove all existing models from the cache
            for f in os.listdir(self._model_cache):
                os.remove(os.path.join(self._model_cache, f))

            self._generate_submodels(self._model_cache,
                                     self._process_count)
        else:
            self._meta_data = self._load_submodels(self._model_cache)
            population = []
            self._generation = 0
            gen_times = []
            while True:
                # check if we need to stop
                if os.path.exists('/home/eey9/pleasestop'):
                    import ipdb
                    ipdb.set_trace()

                logging.info("Generation %d" % self._generation)
                start = datetime.now()

                ir_top = self._immigration_rate_max - \
                    self._immigration_rate
                ir_bottom = self._immigration_rate - \
                    self._immigration_rate

                kr_top = self._keep_rate_max - self._keep_rate
                kr_bottom = self._keep_rate - self._keep_rate_min

                cr_top = self._cross_rate_max - self._cross_rate
                cr_bottom = self._cross_rate - self._cross_rate_min
                # modify the GA rates if needed
                if len(self._RMSE_history) > 30:
                    delta = self._min_delta
                    changes = set(self._RMSE_history[-20:])
                    if len(changes) == 1:
                        # we are stuck in a minima, add immigrants
                        if ir_top >= 2 * delta and \
                           kr_bottom >= delta and \
                           cr_bottom >= delta:
                            # both can be reduced
                            self._immigration_rate += 2 * delta
                        elif ir_top >= delta and (kr_bottom >= delta or
                                                  cr_bottom >= delta):
                            # at least one of them can be reduced
                            self._immigration_rate += delta

                        # adjust the cross and keep values accordingly
                        if kr_bottom >= delta:
                            self._keep_rate -= delta

                        if cr_bottom >= delta:
                            self._cross_rate -= delta

                        if self._mutation_rate < self._mutation_rate_max:
                            self._mutation_rate += delta

                    elif len(changes) > 2:
                        # ok it seems to change every now and again
                        # decrease immigration, let's make GA great again
                        for i in range(2):
                            if ir_bottom >= delta:
                                self._immigration_rate -= delta
                                ir_bottom = self._immigration_rate - \
                                    self._immigration_rate_min

                        if kr_top >= delta:
                            self._keep_rate += delta

                        if cr_top >= delta:
                            self._cross_rate += delta

                        if self._mutation_rate > self._mutation_rate_min:
                            self._mutation_rate -= delta

                    if self._immigration_rate + self._keep_rate + \
                       self._cross_rate != 1.0:
                        self._immigration_rate = 1.0 - (self._keep_rate +
                                                        self._cross_rate)

                    # make sure it all sums to 1
                    try:
                        assert(abs((self._immigration_rate + self._keep_rate +
                               self._cross_rate) - 1.0) < 0.001)
                    except:
                        import ipdb
                        ipdb.set_trace()
                        print("OOPS")

                logging.info("Running generation with:\n")
                logging.info("K: %f, C: %f, I: %f, M: %f" %
                             ((self._keep_rate * population_size),
                              (self._cross_rate * population_size),
                              (self._immigration_rate * population_size),
                              (self._mutation_rate)))

                population = self._run_generation(population_size,
                                                  population,
                                                  self._process_count)
                end = datetime.now() - start
                seconds = end.seconds
                gen_times.append(seconds)
                logging.info("Generation %d lasted %s seconds" %
                             (self._generation, end.seconds))
                logging.info("Mean generation time %f seconds" %
                             (sum(gen_times)/float(len(gen_times))))
                self._generation += 1

    def _run_generation(self, population_size, population, process_count):
        if population == []:
            # first generation - generate new individuals
            logging.info("Generating initial population")
            required_count = population_size - len(population)
            population = self._generate_individuals(required_count, [])

        logging.info("Calculating RMSE")
        population = self._calculate_rmse_mp(population, process_count)

        population.sort(key=lambda x: x.get_rmse())
        logging.info(population[0].get_rmse())
        logging.info([x.get_rmse() for x in population])

        logging.info("Saving population")
        # dump the whole population
        pop_meta = []
        i = 1
        for entry in population:
            CSVFileWriter(os.path.join(self._root_dir,
                                       "population",
                                       "%d.csv" % i),
                          entry.get_instructions())

            pop_meta.append({'name': '%d.csv' % i, 'rmse': entry.get_rmse()})
            i += 1

        CSVFileWriter(os.path.join(self._root_dir, "population", "meta.csv"),
                      pop_meta)

        # append best RMSE to RMSE_history
        self._RMSE_history.append(population[0].get_rmse())

        # keep top 20%
        keep = population[:int(self._keep_rate * (len(population)))]

        # cross between individuals
        for i in range(len(population)):
            population[i].set_probability(self._probabilities[i])

        logging.info("Performing crosses")
        crosses_count = int(self._cross_rate * len(population))
        crosses = self._perform_crosses(population, crosses_count, keep)

        logging.info("Generating immigrants")
        # bring some immigrants as well
        immigrant_count = int(self._immigration_rate * len(population))
        immigrants = self._generate_individuals(immigrant_count,
                                                keep + crosses)

        # merge populations
        all_population = keep + crosses + immigrants
        final_population = []

        # drop any duplicate individuals
        for entry in all_population:
            match = [x for x in final_population if x.same(entry)]
            if len(match) == 0:
                final_population.append(entry)
            else:
                logging.info("Duplicate individual found, removing...")

        # fill in gaps from dropped individuals with newly generated
        # individuals
        while len(final_population) < population_size:
            missing = population_size - len(final_population)
            logging.info("Generating %d individuals to fill in population" %
                         missing)
            individuals = self._generate_individuals(missing, final_population)
            final_population += individuals

        return final_population

    def _calculate_rmse_sp(self, population, n):
        """Here just in case you need to debug"""
        for entry in population:
            entry.get_rmse()

        return population

    def _calculate_rmse_mp(self, population, process_count):
        i = 0
        process_pop = dict()
        while i < len(population):
            for j in range(process_count):
                if str(j) not in process_pop.keys():
                    process_pop[str(j)] = []

                if i < len(population):
                    process_pop[str(j)].append(population[i])
                    i += 1

        final_population = []
        queue = Queue()
        processes = []
        for i in range(process_count):
            pop = process_pop[str(i)]

            process = Process(target=self._calculate_rmse,
                              name="%d" % i,
                              args=(pop, queue))
            process.start()
            processes.append(process)

        for i in range(process_count):
            final_population += queue.get()

        for process in processes:
            process.join()

        return final_population

    def _calculate_rmse(self, population, queue):
        for entry in population:
            entry.get_rmse()

        queue.put(population)

    def _generate_individuals(self, n, current_population):
        individuals = []
        while len(individuals) < n:
            individual = GACustomModel(self._meta_data,
                                       self._data,
                                       self._years)

            new_pop = current_population + individuals
            match = [x for x in new_pop if x.same(individual)]
            if len(match) == 0:
                individuals.append(individual)

        return individuals

    def _perform_crosses(self, population, n, keep):
        individuals = []

        while len(individuals) < n:
            probs = [random.random(), random.random()]
            choices = []
            for prob in probs:
                for entry in population:
                    if entry.get_probability() - prob >= 0:
                        choices.append(entry)
                        break

            if len(choices) != 2:
                logging.info("Choices len %d" % len(choices))
                import ipdb
                ipdb.set_trace()

            individual = choices[0].crossover(choices[1], self._mutation_rate)
            new_pop = keep + individuals
            match = [x for x in new_pop if x.same(individual)]
            if len(match) == 0:
                individuals.append(individual)

        return individuals

    def _generate_submodels(self, model_cache, processes):
        """Generates R submodels"""
        init_vars = NaiveMLProcessModel._INIT_DATA
        instructions = []
        variables = NaiveMLProcessModel._STAGE_ONE +\
            NaiveMLProcessModel._STAGE_TWO

        for variable in variables:
            for method in Gene._METHOD_VALUES:
                other_vars = [x for x in NaiveMLProcessModel._STAGE_ONE if
                              x != variable]
                i = 0
                for i in range(len(other_vars) + 1):
                    choices = combinations(other_vars, i)

                    for choice in choices:
                        train_vars = init_vars + list(choice)
                        instruction = {'name': variable,
                                       'method': method,
                                       'variables': train_vars,
                                       'stage': -999}

                        instructions.append(instruction)

        process_instructions = self._process_split(instructions, processes)

        meta_header = ['name', 'method', 'variables',
                       'stage', 'location', 'year']

        meta_location = os.path.join(model_cache, 'meta.csv')
        CSVFileWriter(meta_location, [], custom_header=meta_header)
        lock = Lock()

        for year in self._years:
            logging.info("Starting year %s" % year)
            new_data = split_dataset(self._data, year)['ml_data']
            # each process_instruction is a list of instructions for that
            # process
            processes = []
            for process_instruction in process_instructions:
                process_name = "Process %d-%d" %\
                               (year,
                                process_instructions.index(
                                        process_instruction))

                process = Process(target=self._train_process,
                                  name=process_name,
                                  args=(new_data,
                                        process_instruction,
                                        model_cache,
                                        process_name,
                                        meta_header,
                                        meta_location,
                                        lock, year))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            logging.info("Finished year %s" % year)

    def _train_process(self, data, instructions, model_cache, pname,
                       meta_header, meta_location, lock, year):
        logging.info("%s starting to train %d submodels" %
                     (pname, len(instructions)))

        for instruction in instructions:
            instruction['year'] = year
            model = MLModel(data, instruction)
            _, fname = mkstemp(dir=model_cache)
            f = open(fname, 'w')
            pickle.dump(model, f)
            f.close()
            logging.info("%s: %d out of %d" %
                         (pname,
                          instructions.index(instruction) + 1,
                          len(instructions)))

            instruction['location'] = fname

        logging.info("%s locking before writing meta data" % pname)
        lock.acquire()
        logging.info("%s writing data" % pname)

        CSVFileWriter(meta_location,
                      instructions,
                      custom_header=meta_header,
                      write_mode='a')

        logging.info("%s releasing lock" % pname)
        lock.release()
        logging.info("%s released lock" % pname)
        logging.info("%s finished" % pname)

    def _process_split(self, entries, processes):
        chunk_size = int(round(len(entries)/float(processes)))
        return [entries[i:i + chunk_size]
                for i in xrange(0, len(entries), chunk_size)]

    def _load_submodels(self, model_cache):
        """Loads meta.csv"""
        reader = CSVFileReader(os.path.join(model_cache, 'meta.csv'))
        content = reader.get_content()

        for entry in content:
            for ch in [',', "'", '[', ']']:
                entry['variables'] = entry['variables'].replace(ch, '')

            entry['variables'] = entry['variables'].split(' ')
            entry['year'] = int(entry['year'])

        return content
