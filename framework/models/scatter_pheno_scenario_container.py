# ScatterPhenoScenarioContainer - container for pheno scenarios
from framework.util.csv_io.csv_filereader import CSVFileReader
from framework.models.scatter_pheno_scenario import from_solution
import random


months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']


class ScatterPhenoScenarioContainer:
    def __init__(self):
        self._population = []
        self._changes = 0

    def contains(self, individual):
        contains = False
        for entry in self._population:
            if entry.same(individual):
                contains = True
                break

        return contains

    def request(self, individual):
        for entry in self._population:
            if entry.same(individual):
                return entry

        raise Exception("Individual not in container")

    def add(self, individual):
        if not self.contains(individual):
            self._population.append(individual)
        else:
            raise ScatterPhenoScenarioContainerException("Individual already "
                                                         "exists")

    def add_container(self, container):
        for entry in container.get_all():
            if not self.contains(entry):
                self.add(entry)

    def len(self):
        return len(self._population)

    def get_all(self):
        return self._population

    def sort(self):
        self._population.sort(key=lambda x: x.get_utility())

    def get(self, index):
        return self._population[index]

    def get_diverse(self, target):
        population = [x for x in self._population if not target.contains(x)]
        population.sort(key=lambda x: (-x.diff(target), x.get_utility()))
        return population[0]

    def index(self, individual):
        if self.contains(individual):
            match = next(x for x in self._population if x.same(individual))
            return self._population.index(match)
        else:
            raise Exception("Individual not in container")

    def replace(self, individual, index):
        if not individual.same(self._population[index]):
            self._changes += 1
            self._population[index] = individual

    def reset_changes(self):
        self._changes = 0

    def get_changes(self):
        return self._changes

    def same(self, container):
        self.sort()
        container.sort()
        result = True
        for x, y in zip(self.get_all(), container.get_all()):
            if not x.same(y):
                result = False
                break

        return result

    def load_file(self, fname, data):
        rows = CSVFileReader(fname).get_content()
        scenarios = []
        for row in rows:
            scenario = []
            keys = [x for x in row.keys()
                    if x not in ['utility', 'rmse', 'cost']]
            for key in keys:
                entry = dict()
                if key in months:
                    entry['month'] = key
                    entry['value'] = (row[key] == "True")
                else:
                    entry['variable'] = key
                    entry['value'] = (row[key] == "True")

                scenario.append(entry)

            final_scenario = from_solution(scenario, data)
            final_scenario.set_rmse(row['rmse'])
            util = float(row['utility'])

            final_scenario.set_utility(util)
            scenarios.append(final_scenario)

        self._population = scenarios

    def sample(self, n):
        contents = []
        while len(contents) < n:
            entry = random.choice(self._population)
            if entry not in contents:
                contents.append(entry)

        container = ScatterPhenoScenarioContainer()
        container._population = contents
        return container


class ScatterPhenoScenarioContainerException(Exception):
    pass
