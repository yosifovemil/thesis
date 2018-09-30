# GAPhenoModel - generates pheno scenario, trains models and calculated RMSE
# from framework.models.ml_process_model import MLProcessModel
from framework.models.pheno_scenario import PhenoScenario
from framework.models.ml_model import MLModel
from framework.models.ml_process_model import MLProcessModel
from framework.models.naive_ml_process_model_memfix \
    import NaiveMLProcessModelMemfix
import framework.data.formats as formats
from copy import deepcopy
from framework.models.ml_train_exception import MLTrainException
import logging


class GAPhenoModel:
    def __init__(self, data, variable, methods, max_rmse, max_cost, base,
                 pheno_scenario=None):
        self._value = None
        self._data = deepcopy(data)
        self._methods = methods
        self._max_rmse = max_rmse
        self._max_cost = max_cost
        self._years = set([x[formats.DATE].year for x in self._data['ml_data']
                           if x[formats.DW_PLANT] is not None])
        self._variable = variable
        self._base = base

        if pheno_scenario is None:
            self._generate_scenario()
        else:
            self._pheno_scenario = pheno_scenario
            try:
                self.calc_value()
            except MLTrainException:
                logging.warn("Cross unsuccessful, generating new individual")
                self._generate_scenario()

    def _generate_scenario(self):
        # get weeks
        weeks = sorted(list(set([int(x[formats.DATE].strftime("%W")) + 1
                                 for x in self._data['ml_data']])))

        scenario_kwargs = {'week': weeks,
                           'variable': self._variable}

        scenario_ok = False
        while not scenario_ok:
            scenario_viable = False
            while not scenario_viable:
                self._pheno_scenario = PhenoScenario(**scenario_kwargs)
                scenario_viable = self._pheno_scenario.viable(self._data)

            try:
                self.calc_value()
                scenario_ok = True
            except MLTrainException:
                pass

    def cross(self, pheno_model):
        viable = False
        while not viable:
            pheno_scenario = self._pheno_scenario.cross(
                                pheno_model._pheno_scenario)

            viable = pheno_scenario.viable(self._data)

        return GAPhenoModel(self._data, self._variable,
                            self._methods, self._max_rmse, self._max_cost,
                            self._base, pheno_scenario)

    def get_pheno_scenario(self):
        return self._pheno_scenario

    def get_value(self):
        if self._value is None:
            self.calc_value()

        if self._pheno_scenario.get_cost() > self._max_cost:
            return 1000 * abs(self._value)
        else:
            return self._value

    def get_rep_values(self):
        return self._rep_values

    def get_reps(self):
        return self._reps

    def calc_value(self):
        if self._value is None:
            self._data_subset = self._pheno_scenario.apply(self._data)
            self._reps = self.train_models()

            self._rep_values = []
            for rep in self._reps:
                # get the mean rmse of all models
                value = sum([x.get_rmse() for x in rep])/len(rep)
                self._rep_values.append(value)

            self._value = sum(self._rep_values)/len(self._rep_values)

        return self._value

    def same(self, model):
        return self._pheno_scenario.same(model.get_pheno_scenario())

    def train_models(self):
        # UNDER CONSTRUCTION TODO ADD MORE MODELS
        reps = []
        for i in range(10):
            models = []
            for method in self._methods:
                mlp_methods = MLModel._METHODS + \
                              [NaiveMLProcessModelMemfix._NAME]
                if method in mlp_methods:
                    # train simple ML models or naive ml process model
                    model = MLProcessModel(method,
                                           self._data,
                                           self._data_subset,
                                           self._get_variables(),
                                           self._base)

                    model.set_max_rmse(self._max_rmse[method])
                    models.append(model)

            reps.append(models)

        return reps

    def _get_variables(self):
        """Returns a list of variables to be used for building the model"""
        return [x.get_name() for x in self._pheno_scenario.get_genes()
                if x.get_type() == 'variable' and x.get_value()]

    def __repr__(self):
        rep = ""
        rep += self._pheno_scenario.__repr__()
        rep += "\n"

        return rep
