# MLProcessModel - a class used for building generic process ML models
# or simple ML models. Uses MLModelMemfix to build a model with or without
# submodels, given a set of instructions. This class can create the following
# models: simple ML models, Naive ML process model (2 stages) and
# the best GA ML process model.

# IMPORTANTE: Performs year-based cross validation

from framework.models.ml_model_memfix import MLModelMemfix
import framework.data.formats as formats
from framework.models.ml_train_exception import MLTrainException
from framework.models.naive_ml_process_model_memfix \
        import NaiveMLProcessModelMemfix
from framework.scenarios.BSBEC.cd_model_trainer import CDModelTrainer
from framework.models.GA_win_model import GAWinModel


class MLProcessModel:
    def __init__(self, method, data, data_sub, variables, base):
        self._data = data  # full data for testing models
        self._data_sub = data_sub   # for training models
        self._method = method
        self._variables = variables
        self._base = base

        self._instructions = self._generate_instructions()
        self._predictions = self.predict()

    def get_rmse(self):
        return self._rmse

    def get_normalised_rmse(self):
        if self._method == MLModelMemfix._GBM:
            rat = 1.168614
        elif self._method == MLModelMemfix._RANDOM_FOREST:
            rat = 1.149370
        elif self._method == MLModelMemfix._KNN:
            rat = 1.369238
        elif self._method == NaiveMLProcessModelMemfix._NAME:
            rat = 1.261662
        elif self._method == GAWinModel._NAME:
            rat = 1.599346
        else:
            raise Exception("Unknown method ratio")

        x = self._rmse_abs
        max_x = 1.4 * self._max_rmse
        min_x = self._max_rmse / rat

        return (x - min_x)/(max_x - min_x)

    def get_absolute_rmse(self):
        return self._rmse_abs

    def get_method(self):
        return self._method

    def set_max_rmse(self, max_rmse):
        self._max_rmse = max_rmse
        self._rmse = (self._rmse_abs - self._max_rmse) / self._max_rmse

    def predict(self):
        """This is where cross validation happens"""
        predictions = []

        if self._method in MLModelMemfix._METHODS:
            model_data = {'train_data': self._data_sub['train_data'],
                          'test_data': self._data['test_data']}

            # We often get Exceptions here when new code executes this
            # class, so leaving that here
            try:
                model = MLModelMemfix(model_data, self._instructions,
                                      self._base)
            except MLTrainException:
                raise
            except Exception as e:
                import ipdb
                ipdb.set_trace()
                print e.message
                print "Found the error!"

            predictions += model.get_predictions()
        elif self._method == NaiveMLProcessModelMemfix._NAME:
            stage_one = {'train_data': self._data_sub['train_data'],
                         'test_data': self._data['test_data']}

            stage_two_train = [x for x in self._data_sub['train_data']
                               if x[formats.DW_PLANT] is not None]
            stage_two_test = [x for x in self._data['test_data']
                              if x[formats.DW_PLANT] is not None]
            stage_two = {'train_data': stage_two_train,
                         'test_data': stage_two_test}

            model_data = {'stage_one': stage_one,
                          'stage_two': stage_two}

            model = NaiveMLProcessModelMemfix(model_data, "cumul_reduce")
            predictions = model.get_predictions(compare=True)
            for entry in predictions:
                entry['real'] = entry[formats.DW_PLANT]

        elif self._method == GAWinModel._NAME:
            stage_one = {'train_data': self._data_sub['train_data'],
                         'test_data': self._data['test_data']}

            stage_two_train = [x for x in self._data_sub['train_data']
                               if x[formats.DW_PLANT] is not None]
            stage_two_test = [x for x in self._data['test_data']
                              if x[formats.DW_PLANT] is not None]
            stage_two = {'train_data': stage_two_train,
                         'test_data': stage_two_test}

            model_data = {'stage_one': stage_one,
                          'stage_two': stage_two}

            model = GAWinModel(model_data)
            predictions = model.get_predictions(compare=True)
            for entry in predictions:
                entry['real'] = entry[formats.DW_PLANT]

        return predictions

    def _generate_instructions(self):
        if self._method in [MLModelMemfix._RANDOM_FOREST, MLModelMemfix._KNN,
                            MLModelMemfix._GBM]:
            # there will be only one instruction
            instruction = {'name': formats.DW_PLANT,
                           'method': self._method,
                           'variables': self._variables,
                           'check_num': ''}

        elif self._method in [NaiveMLProcessModelMemfix._NAME,
                              GAWinModel._NAME]:
            # NaiveMLProcessModelMemfix generates them itself
            # this stops only the NaiveMLProcess models, it does not
            # influence the normal ML models
            return None

        elif self._method == CDModelTrainer.NAME:
            return None

        return instruction

    def __repr__(self):
        rep = "%s %f\n" % (self._method, self._rmse)

        return rep
