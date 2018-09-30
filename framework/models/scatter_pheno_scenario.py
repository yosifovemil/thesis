# ScatterPhenoScenario - data structure that defines which variables/months
# should be used for a particular ScatterPhenoModel
import framework.data.formats as formats
from copy import deepcopy
from framework.util.config_parser import SpaceAwareConfigParser
import os
from framework.models.naive_ml_process_model_memfix \
    import NaiveMLProcessModelMemfix


# alternative constructor
def from_solution(solution, data):
    months = []
    variables = []

    for entry in solution:
        if 'month' in entry.keys():
            months.append(entry)
        elif 'variable' in entry.keys():
            variables.append(entry)
        else:
            raise Exception("Unhandled entry")

    return ScatterPhenoScenario(months, variables, data)


class ScatterPhenoScenario:
    def __init__(self, months, variables, data):
        """months - [{'month': 'August', 'value': True/False}]
        variables - [{'variable': 'stem_count', 'value': True/False}]
        """
        self._all_months = sorted(months, key=lambda x: x['month'])
        self._all_variables = sorted(variables, key=lambda x: x['variable'])

        self._solution = self._all_months + self._all_variables
        self._utility = None
        self._rmse = None
        self._calc_cost()
        self._data = data

    def diff(self, input_):
        if input_.__class__ == ScatterPhenoScenario:
            return self._diff_individual(input_)
        else:
            # it has to be ScatterPhenoScenarioContainer
            population = input_.get_all()
            diff = 0
            for entry in population:
                diff += self._diff_individual(entry)

            return diff

    def _diff_individual(self, pheno_scenario):
        diff = 0
        for val1, val2 in zip(self._solution, pheno_scenario.get_solution()):
            if val1['value'] != val2['value']:
                diff += 1

        return diff

    def is_evaluated(self):
        if self._rmse is None or self._utility is None:
            return False

        return True

    def get_months(self):
        return [x['month'] for x in self._all_months if x['value']]

    def get_variables(self):
        return [x['variable'] for x in self._all_variables if x['value']]

    def get_solution(self):
        return self._solution

    def get_cost(self, absolute=False):
        if absolute:
            return self._scenario_cost
        else:
            return self._scenario_cost/self._max_cost

    def set_utility(self, utility):
        if self.is_evaluated():
            raise Exception("Utility already set")

        self._utility = utility

    def has_utility(self):
        if self._utility is None:
            return False

        return True

    def get_utility(self):
        if not self.has_utility():
            raise Exception("Pheno scenario unevaluated")

        return self._utility

    def set_rmse(self, rmse):
        if self.is_evaluated():
            raise Exception("RMSE already set")

        self._rmse = rmse

    def get(self, index):
        return self._solution[index]['value']

    def same(self, scenario):
        sol1 = self.get_solution()
        sol2 = scenario.get_solution()
        same = True

        if len(sol1) != len(sol2):
            raise Exception("Solutions have different lengths")

        for e1, e2 in zip(sol1, sol2):
            # check if we are comparing the same variable
            for key in ['month', 'variable']:
                if key in e1.keys():
                    if key not in e2.keys() or e1[key] != e2[key]:
                        raise Exception("Variable order is wrong!")

            if e1['value'] != e2['value']:
                same = False
                break

        return same

    def apply(self, data):
        final_data = deepcopy(data)
        for key in final_data.keys():
            # subset the months
            new_data = [x for x in final_data[key] if
                        x[formats.DATE].strftime("%B")
                        in self.get_months()]

            for entry in new_data:
                entry_keys = [x for x in entry.keys()
                              if x not in self.get_variables()]

                # always keep date and dry weight
                entry_keys.remove(formats.DATE)
                entry_keys.remove(formats.DW_PLANT)
                for entry_key in entry_keys:
                    del entry[entry_key]

            final_data[key] = new_data

        return final_data

    def toggle(self, index):
        self._solution[index]['value'] = not self._solution[index]['value']

    def valid(self, process=False, debug=False):
        """Process is false for simple_ml and true for compound models"""
        if debug:
            import ipdb
            ipdb.set_trace()

        if len(self.get_months()) < 3 or len(self.get_variables()) < 3:
            return False

        # check if we can get yield data for each year
        # otherwise the scenario might subset the data in a way that makes
        # 2011, 2015 cross validation impossible
        data_sub = self.apply(self._data)
        for year in [2011, 2015]:
            match = [x for x in data_sub['ml_data']
                     if x[formats.DW_PLANT] is not None and
                     x[formats.DATE].year == year]

            if len(match) == 0:
                return False

        if process:
            # make sure there is at least one variable from INIT_DATA
            # in the training dataset
            variables = self.get_variables()
            if len([x for x in variables
                    if x in NaiveMLProcessModelMemfix._INIT_DATA]) == 0:
                return False

        return True

    def _calc_cost(self):
        config = SpaceAwareConfigParser()
        config.read(os.path.expandvars("$ALF_CONFIG/pheno_scenario.ini"))

        max_cost = 0.0  # the maximum a scenario could cost
        scenario_cost = 0.0  # actual cost of this scenario
        for var in self._all_variables:
            cost = float(config.get('cost', var['variable']))
            max_cost += cost
            if var['value']:
                scenario_cost += cost

        max_cost *= len(self._all_months)
        scenario_cost *= len([x for x in self._all_months if x['value']])

        self._max_cost = max_cost
        self._scenario_cost = scenario_cost

    def __repr__(self):
        rep = ""
        # for month in self._all_months:
        # rep += "%s\t" % month['month']

        # for variable in self._all_variables:
        # rep += "%s\t" % variable['variable']

        # rep += "\n"
        for month in self._all_months:
            rep += "%s\t" % month['value']

        for variable in self._all_variables:
            rep += "%s\t" % variable['value']

        return rep

    def to_dict(self):
        result = dict()
        for entry in self._all_months:
            result[entry['month']] = entry['value']

        for entry in self._all_variables:
            result[entry['variable']] = entry['value']

        result['rmse'] = self._rmse
        result['cost'] = self.get_cost()
        result['utility'] = self._utility

        return result

    def copy_to(self, target):
        target.set_utility(self.get_utility())
        target.set_rmse(self._rmse)
