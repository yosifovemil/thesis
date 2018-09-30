# ScatterPhenoModel - takes a PhenoScenario and evaluates objective function
# This is the compound version of ScatterPhenoModel

from framework.models.ml_process_model import MLProcessModel


class ScatterPhenoModel:

    def __init__(self, data, base, pheno_scenario, methods, max_rmse):
        """pheno_scenario - ScatterPhenoScenario object"""
        self._data = data
        self._base = base
        self._methods = methods
        self._pheno_scenario = pheno_scenario
        self._max_rmse = max_rmse

        self._data_sub = pheno_scenario.apply(data)
        self._train()

    def get_rmse(self):
        return (sum([x.get_normalised_rmse() for x in self._models]) /
                len(self._models))

    def get_absolute_rmse(self):
        """Example return format: {'knn': 300, 'rf': 340}"""
        result = dict()
        for model in self._models:
            result[model.get_method()] = model.get_absolute_rmse()

        return result

    def _train(self):
        self._models = []
        for method in self._methods:
            model = MLProcessModel(method, self._data, self._data_sub,
                                   self._pheno_scenario.get_variables(),
                                   self._base)
            model.set_max_rmse(self._max_rmse[method])
            self._models.append(model)
