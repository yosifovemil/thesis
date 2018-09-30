# ScatterPhenoScenarioBuilder - builds a scenario using one of the three
# generation methods, or combines scenarios using one of the seven
# combination methods.
import framework.data.formats as formats
from framework.models.scatter_pheno_scenario import ScatterPhenoScenario, \
        from_solution
from framework.models.scatter_pheno_scenario_container \
        import ScatterPhenoScenarioContainer
import random
from copy import deepcopy
from framework.util.csv_io.csv_filereader import CSVFileReader


class ScatterPhenoScenarioBuilder:
    def __init__(self, data, variables, process=False):
        """process is false for simple_ml and true for process models"""
        self._months = sorted(list(set([x[formats.DATE].strftime("%B")
                                       for x in data['ml_data']])))
        self._variables = variables
        self._n = len(self._variables + self._months)
        self._data = data
        self._build_combine_functions()
        self._last_cm = None
        self._process = process

    def g1(self, count):
        done = False
        iterations = 0
        while not done:
            solutions = ScatterPhenoScenarioContainer()
            seed_solution = self._random_solution()
            solutions.add(seed_solution)

            h_max = self._n

            # maximum number of solutions is h_max - 1
            if count > h_max - 1:
                raise Exception("Could not generate %d solutions with "
                                "G1 and h_max %d" % (count, h_max))

            for h in range(2, h_max):
                new_solution = deepcopy(seed_solution)
                index = 0
                while index < self._n:
                    new_solution.toggle(index)
                    index += h

                if solutions.len() < count and \
                        new_solution.valid(process=self._process):
                    solutions.add(new_solution)
                else:
                    break

            if solutions.len() == count:
                done = True
            else:
                iterations += 1

            if iterations > 10:
                raise Exception("Could not generate enough valid solutions")

        return solutions

    def g2(self, score_table):
        return self._build_g_solution(score_table, False)

    def g3(self, score_table):
        return self._build_g_solution(score_table, True)

    def flip(self, scenario, score_entry):
        """Builds new solution where the variable in score_entry is flipped"""
        months = deepcopy(scenario._all_months)
        variables = deepcopy(scenario._all_variables)

        if score_entry['type'] == 'month':
            match = next(x for x in months
                         if x['month'] == score_entry['name'])
        elif score_entry['type'] == 'variable':
            match = next(x for x in variables
                         if x['variable'] == score_entry['name'])
        else:
            raise Exception("Unhandled type %s" % score_entry['type'])

        match['value'] = not match['value']

        new_scenario = ScatterPhenoScenario(months, variables, self._data)
        return new_scenario

    def swap(self, scenario, index, index2):
        """Builds new solution where the values
        on index and index2 are swapped
        """
        months = deepcopy(scenario._all_months)
        variables = deepcopy(scenario._all_variables)

        solution = months + variables
        temp = solution[index]['value']
        solution[index]['value'] = solution[index2]['value']
        solution[index2]['value'] = temp

        new_scenario = ScatterPhenoScenario(months, variables, self._data)
        return new_scenario

    def combine(self, scenario1, scenario2, score_table):
        # choose function to use
        prob = random.random()
        for f in self._cm_functions:
            if prob < f['probability']:
                self._last_cm = f
                break

        return(self._last_cm['function'](scenario1, scenario2, score_table))

    def success(self, so_much_win):
        self._last_cm['success'] = (0.3 * self._last_cm['success'] +
                                    0.7 * so_much_win)

        self._update_cm_functions()

    def _update_cm_functions(self):
        # update probabilities
        self._cm_functions.sort(key=lambda x: x['method'])
        last_prob = 0
        total = sum([x['success'] for x in self._cm_functions])
        for entry in self._cm_functions:
            entry['probability'] = (float(entry['success'])/total) + last_prob
            last_prob = entry['probability']

        self._cm_functions[-1]['probability'] = 1.0

    def _build_g_solution(self, score_table, default_value):
        valid = False
        while not valid:
            x = self._hardcoded_solution(default_value)
            x = self._flip_by_prob(x, score_table)
            valid = x.valid(process=self._process)

        return x

    def _random_solution(self):
        valid = False
        while not valid:
            months = []
            for month in self._months:
                months.append({'month': month,
                              'value': bool(random.randint(0, 1))})

            s_vars = []
            for var in self._variables:
                s_vars.append({'variable': var,
                               'value': bool(random.randint(0, 1))})

            solution = ScatterPhenoScenario(months, s_vars, self._data)
            valid = solution.valid(process=self._process)

        return solution

    def _hardcoded_solution(self, value):
        months = []
        for month in self._months:
            months.append({'month': month,
                          'value': value})

        s_vars = []
        for var in self._variables:
            s_vars.append({'variable': var,
                           'value': value})

        solution = ScatterPhenoScenario(months, s_vars, self._data)

        return solution

    def _flip_by_prob(self, individual, score_table):
        for entry in individual.get_solution():
            t = next(x for x in entry.keys() if x != 'value')
            match = next(x for x in score_table
                         if x['type'] == t and
                         x['name'] == entry[t] and
                         x['value'] != entry['value'])

            prob = 1 - match['score']
            if random.random() <= prob:
                entry['value'] = match['value']

        return individual

    def _and_solution(self, scenario1, scenario2):
        new_solution = []
        for var1, var2 in zip(scenario1.get_solution(),
                              scenario2.get_solution()):
            new_var = deepcopy(var1)
            new_var['value'] = var1['value'] and var2['value']
            new_solution.append(new_var)

        return from_solution(new_solution, self._data)

    def _cm1(self, scenario1, scenario2, score_table):
        scenario = self._and_solution(scenario1, scenario2)
        solution = scenario.get_solution()

        safe_solution = self._make_valid(deepcopy(solution))

        # loop through the true vars and try to turn some of them to false
        for entry in [x for x in solution if x['value']]:
            t = next(x for x in entry.keys() if x != 'value')
            match = next(x for x in score_table
                         if x['type'] == t and
                         x['name'] == entry[t] and
                         x['value'] == entry['value'])
            prob = 1 - match['score']

            if random.random() <= prob:
                entry['value'] = False
                test_scenario = from_solution(solution, self._data)
                if test_scenario.valid(process=self._process):
                    # save the solution if it is still viable
                    safe_solution = deepcopy(solution)
                else:
                    break

        final_solution = from_solution(safe_solution, self._data)

        if not final_solution.valid(process=self._process):
            raise Exception("Solution not valid")

        return final_solution

    def _cm2(self, scenario1, scenario2, score_table):
        scenario = self._and_solution(scenario1, scenario2)
        solution = scenario.get_solution()

        safe_solution = self._make_valid(deepcopy(solution))

        # loop through the true vars and try to turn some of them to false
        valid = True
        while valid:
            solution = deepcopy(safe_solution)
            candidates = [x for x in solution if x['value']]

            entry = random.choice(candidates)
            entry['value'] = False

            test_scenario = from_solution(solution, self._data)
            valid = test_scenario.valid(process=self._process)
            if valid:
                # save the solution if it is still viable
                safe_solution = deepcopy(solution)

        final_solution = from_solution(safe_solution, self._data)

        if not final_solution.valid(process=self._process):
            raise Exception("Solution not valid")

        return final_solution

    def _cm3(self, scenario1, scenario2, score_table):
        new_solution = []

        for var1, var2 in zip(scenario1.get_solution(),
                              scenario2.get_solution()):
            entry = dict()
            t = next(x for x in var1.keys() if x != 'value')
            entry[t] = var1[t]

            weight = self._var_weight(scenario1, scenario2, var1, var2)

            if random.random() <= weight:
                entry['value'] = True
            else:
                entry['value'] = False

            new_solution.append(entry)

        new_solution = self._make_valid(new_solution)
        final_solution = from_solution(new_solution, self._data)
        if not final_solution.valid(process=self._process):
            raise Exception("Solution not valid")

        return final_solution

    def _cm4(self, scenario1, scenario2, score_table):
        new_solution = self._and_solution(scenario1, scenario2).get_solution()

        for var in [x for x in new_solution if not x['value']]:
            index = new_solution.index(var)
            var1 = scenario1.get_solution()[index]
            var2 = scenario2.get_solution()[index]

            weight = self._var_weight(scenario1, scenario2, var1, var2)

            if random.random() < weight:
                var['value'] = True

        new_solution = self._make_valid(new_solution)
        final_solution = from_solution(new_solution, self._data)
        if not final_solution.valid(process=self._process):
            raise Exception("Solution not valid")

        return final_solution

    def _cm5(self, scenario1, scenario2, score_table):
        new_solution = self._and_solution(scenario1, scenario2).get_solution()

        for var in [x for x in new_solution if not x['value']]:
            if random.random() < 0.5:
                var['value'] = True

        new_solution = self._make_valid(new_solution)
        final_solution = from_solution(new_solution, self._data)
        if not final_solution.valid(process=self._process):
            raise Exception("Solution not valid")

        return final_solution

    def _cm6(self, scenario1, scenario2, score_table):
        new_scenario = self._build_g_solution(score_table, False)
        new_solution = new_scenario.get_solution()

        for new_var, var1, var2 in zip(new_solution,
                                       scenario1.get_solution(),
                                       scenario2.get_solution()):
            if not (var1['value'] or var2['value']):
                new_var['value'] = False

        new_solution = self._make_valid(new_solution)
        final_solution = from_solution(new_solution, self._data)
        if not final_solution.valid(process=self._process):
            raise Exception("Solution not valid")

        return final_solution

    def _var_weight(self, scenario1, scenario2, var1, var2):
        util1 = scenario1.get_utility()
        util2 = scenario2.get_utility()

        weight = 1 - (((util1 * int(var1['value'])) +
                      (util2 * int(var2['value']))) / (util1 + util2))

        return weight

    def _make_valid(self, solution):
        while not from_solution(solution,
                                self._data).valid(process=self._process):
            months = len([x for x in solution
                         if 'month' in x.keys() and x['value']])
            variables = len([x for x in solution
                             if 'variable' in x.keys() and x['value']])
            changed = False
            if months < 3:
                solution = self._flip_random(solution, 'month', False)
                changed = True

            if variables < 3:
                solution = self._flip_random(solution, 'variable', False)
                changed = True

            if not changed:
                new_solution = deepcopy(solution)
                candidate_months = [x for x in new_solution
                                    if 'month' in x.keys()
                                    and not x['value']]

                # check if switching any particular month makes the
                # solution viable
                for month in candidate_months:
                    month['value'] = True
                    if from_solution(new_solution,
                                     self._data).valid(process=self._process):
                        changed = True
                    else:
                        month['value'] = False

                # just pick one and go around the loop again
                if not changed and len(candidate_months) > 0:
                    month = random.choice(candidate_months)
                    month['value'] = True
                else:
                    # no candidate months - definitely try to change
                    # a variable to True
                    solution = self._flip_random(solution, 'variable', False)
                    changed = True

                solution = new_solution

        return solution

    def _flip_random(self, solution, t, orig_val):
        entry = random.choice([x for x in solution
                               if t in x.keys() and x['value'] == orig_val])
        entry['value'] = not orig_val
        return solution

    def _build_combine_functions(self):
        default_success = 1
        self._cm_functions = [{'method': 'CM1',
                               'function': self._cm1,
                               'success': default_success,
                               'probability': 0},
                              {'method': 'CM2',
                               'function': self._cm2,
                               'success': default_success,
                               'probability': 0},
                              {'method': 'CM3',
                               'function': self._cm3,
                               'success': default_success,
                               'probability': 0},
                              {'method': 'CM4',
                               'function': self._cm4,
                               'success': default_success,
                               'probability': 0},
                              {'method': 'CM5',
                               'function': self._cm5,
                               'success': default_success,
                               'probability': 0},
                              {'method': 'CM6',
                               'function': self._cm6,
                               'success': default_success,
                               'probability': 0}]
        self._update_cm_functions()

    def load_cm(self, fname):
        content = CSVFileReader(fname).get_content()
        for entry in content:
            entry['probability'] = float(entry['probability'])
            entry['success'] = float(entry['success'])
            if entry['method'] == "CM1":
                entry['function'] = self._cm1
            elif entry['method'] == "CM2":
                entry['function'] = self._cm2
            elif entry['method'] == "CM3":
                entry['function'] = self._cm3
            elif entry['method'] == "CM4":
                entry['function'] = self._cm4
            elif entry['method'] == "CM5":
                entry['function'] = self._cm5
            elif entry['method'] == "CM6":
                entry['function'] = self._cm6

        self._cm_functions = content
